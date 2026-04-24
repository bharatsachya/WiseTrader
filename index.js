import { logger } from "src/logger/logger";
import { config } from "src/config/config";
import { ConfigKey } from "src/config/keys";
import {
  ExchangeRequest,
  DisconnectRequest,
  ProviderConnectionStatus,
  ProviderType,
  OAuthTokenSecret,
  ProviderRepo,
} from "@/models/providerConnection";
import { ApiResponse } from "src/models";
import { returnPayload } from "src/utils/response";
import { getProviderConnectionsCollection, getUsersCollection } from "src/utils/dbUtils";
import { UserType } from "src/models/user/user";
import { awsSecretsManagerService } from "src/services/awsSecretService";

export class TokenExpiredError extends Error {
  constructor(message: string = "Token expired or revoked") {
    super(message);
    this.name = "TokenExpiredError";
  }
}

export class WorkspaceFetchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WorkspaceFetchError";
  }
}

export class ProviderService {
  private getSecretName(orgId: string, provider: ProviderType): string {
    return `${orgId}/provider-oauth/${provider}`;
  }

  private getUserSecretName(userId: string, provider: ProviderType): string {
    return `users/${userId}/${provider}-oauth`;
  }

  private async fetchWithRetry(url: string, token?: string, options: RequestInit = {}): Promise<Response> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      "User-Agent": "ByteBell-Admin-Backend",
      ...((options.headers as Record<string, string>) || {}),
    };

    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const res = await fetch(url, { ...options, headers });
        return res;
      } catch (err) {
        if (attempt === 3) throw err;
        const delay = 200 * 2 ** (attempt - 1);
        logger.warn(`⚠️ Fetch failed for ${url} (attempt ${attempt}/3), retrying in ${delay}ms...`);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
    throw new Error(`Fetch failed after 3 attempts: ${url}`);
  }

  async markRevoked(orgId: string, provider: ProviderType, userId?: string): Promise<void> {
    const scope = userId ? `user ${userId}` : `org ${orgId}`;
    logger.warn(`⚠️ Marking ${provider} token as revoked for ${scope} — cleaning up`);
    try {
      const secretName = userId ? this.getUserSecretName(userId, provider) : this.getSecretName(orgId, provider);
      await awsSecretsManagerService.deleteSecretByName(secretName);

      if (!userId) {
        const collection = await getProviderConnectionsCollection();
        await collection.deleteOne({ org_id: orgId, provider });
      }
    } catch (err) {
      logger.error(`❌ Failed to clean up revoked ${provider} token for ${scope}:`, err);
    }
  }

  private async refreshAccessToken(orgId: string, provider: ProviderType, userId?: string): Promise<string | null> {
    const secretName = userId ? this.getUserSecretName(userId, provider) : this.getSecretName(orgId, provider);
    const scope = userId ? `user ${userId}` : `org ${orgId}`;

    try {
      const secretString = await awsSecretsManagerService.getSecretString(secretName);
      if (!secretString) {
        return null;
      }

      const secret = JSON.parse(secretString) as OAuthTokenSecret;
      if (!secret.refresh_token) {
        logger.warn(`⚠️ No refresh_token stored for ${provider} (${scope})`);
        return null;
      }

      const authProxyUrl = config.get(ConfigKey.AUTH_PROXY_URL);
      const tokenRes = await fetch(`${authProxyUrl}/api/refresh-token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: secret.refresh_token, provider, org_id: orgId }),
      });

      if (!tokenRes.ok) {
        const errData = (await tokenRes.json()) as { error?: string };
        logger.error(`❌ Auth-proxy token refresh failed (${tokenRes.status}) for ${scope}: ${errData.error}`);
        return null;
      }

      const tokenData = (await tokenRes.json()) as {
        access_token: string;
        refresh_token?: string;
        expires_at?: string;
      };

      const newExpiresAt = tokenData.expires_at ?? secret.expires_at;

      const updatedSecret: OAuthTokenSecret = {
        ...secret,
        access_token: tokenData.access_token,
        refresh_token: tokenData.refresh_token ?? secret.refresh_token,
        expires_at: newExpiresAt,
      };

      await awsSecretsManagerService.upsertSecret(secretName, JSON.stringify(updatedSecret));

      // Update MongoDB expires_at (only for org-level connections)
      if (!userId) {
        const collection = await getProviderConnectionsCollection();
        await collection.updateOne(
          { org_id: orgId, provider },
          { $set: { expires_at: newExpiresAt, updated_at: new Date().toISOString() } },
        );
      }

      logger.info(`✅ Refreshed ${provider} token for ${scope}`);
      return tokenData.access_token;
    } catch (err) {
      logger.error(`❌ Error refreshing ${provider} token for ${scope}:`, err);
      return null;
    }
  }

  private async getValidAccessToken(orgId: string, provider: ProviderType, userId?: string): Promise<string | null> {
    const secretName = userId ? this.getUserSecretName(userId, provider) : this.getSecretName(orgId, provider);

    try {
      let expiresAt: string | null = null;

      if (!userId) {
        const collection = await getProviderConnectionsCollection();
        const connection = await collection.findOne({ org_id: orgId, provider });
        if (connection) expiresAt = connection.expires_at;
      }

      const secretString = await awsSecretsManagerService.getSecretString(secretName);
      if (!secretString) return null;

      const secret = JSON.parse(secretString) as OAuthTokenSecret;
      if (!expiresAt && secret.expires_at) expiresAt = secret.expires_at;

      // Check expiration
      if (expiresAt && new Date(expiresAt) < new Date()) {
        logger.info(`⏳ Token expired for ${provider} (${userId ? `user ${userId}` : `org ${orgId}`}), refreshing...`);
        return this.refreshAccessToken(orgId, provider, userId);
      }

      return secret.access_token;
    } catch (err) {
      logger.error(`❌ Error getting valid access token for ${provider}:`, err);
      return null;
    }
  }

  async claimTicket(body: ExchangeRequest, userId?: string): Promise<ApiResponse<ProviderConnectionStatus>> {
    const { ticket_id, org_id, provider } = body;

    try {
      // 1. Claim ticket from auth-proxy
      const authProxyUrl = config.get(ConfigKey.AUTH_PROXY_URL);
      const ticketRes = await this.fetchWithRetry(`${authProxyUrl}/api/exchange-ticket`, undefined, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticket_id }),
      });

      if (!ticketRes.ok) {
        if (ticketRes.status === 404 || ticketRes.status === 410) {
          return returnPayload(false, "This connection link has expired or has already been used. Please try again.", {
            connected: false,
            username: "",
            avatarUrl: "",
            expiresAt: null,
          });
        }
        const errData = (await ticketRes.json()) as { error?: string };
        return returnPayload(false, errData.error || "Failed to claim ticket", {
          connected: false,
          username: "",
          avatarUrl: "",
          expiresAt: null,
        });
      }

      const ticket = (await ticketRes.json()) as {
        access_token: string;
        refresh_token?: string;
        expires_at?: string;
      };

      const accessToken = ticket.access_token;
      const refreshToken = ticket.refresh_token;
      const expiresAt = ticket.expires_at || null;

      let username = "";
      let avatarUrl = "";

      if (provider === "github") {
        const userRes = await this.fetchWithRetry("https://api.github.com/user", accessToken);
        const userData = (await userRes.json()) as { login: string; avatar_url: string };
        username = userData.login;
        avatarUrl = userData.avatar_url;
      } else if (provider === "bitbucket") {
        const userRes = await this.fetchWithRetry("https://api.bitbucket.org/2.0/user", accessToken);
        const userData = (await userRes.json()) as { username: string; links?: { avatar?: { href: string } } };
        username = userData.username;
        avatarUrl = userData.links?.avatar?.href || "";
      } else {
        return returnPayload(false, `Unsupported provider: ${provider}`, {
          connected: false,
          username: "",
          avatarUrl: "",
          expiresAt: null,
        });
      }

      // 3. Store in AWS Secrets Manager
      const secretName = this.getSecretName(org_id, provider);
      const secretValue: OAuthTokenSecret = {
        access_token: accessToken,
        refresh_token: refreshToken,
        expires_at: expiresAt || undefined,
        provider,
        created_at: new Date().toISOString(),
      };

      await awsSecretsManagerService.upsertSecret(
        secretName,
        JSON.stringify(secretValue),
        `OAuth token for ${provider} - ${username}`,
      );

      // 4. Upsert MongoDB connection doc
      const collection = await getProviderConnectionsCollection();
      await collection.updateOne(
        { org_id, provider, ...(userId ? { user_id: userId } : {}) },
        {
          $set: {
            username,
            avatar_url: avatarUrl,
            expires_at: expiresAt,
            aws_secret_name: secretName,
            updated_at: new Date().toISOString(),
            ...(userId ? { user_id: userId } : {}),
          },
          $setOnInsert: {
            created_at: new Date().toISOString(),
          },
        },
        { upsert: true },
      );

      return returnPayload(true, "Connected successfully", {
        connected: true,
        username,
        avatarUrl,
        expiresAt,
      });
    } catch (error: unknown) {
      logger.error(`❌ Error in claimTicket (${provider}):`, error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      return returnPayload(false, `Failed to claim ticket: ${errorMessage}`, {
        connected: false,
        username: "",
        avatarUrl: "",
        expiresAt: null,
      });
    }
  }

  async getStatus(
    orgId: string,
    userId?: string,
  ): Promise<ApiResponse<{ github: ProviderConnectionStatus | null; bitbucket: ProviderConnectionStatus | null }>> {
    try {
      const status: { github: ProviderConnectionStatus | null; bitbucket: ProviderConnectionStatus | null } = {
        github: null,
        bitbucket: null,
      };

      const collection = await getProviderConnectionsCollection();

      // 1. Fetch organization-level connections
      const orgDocs = await collection.find({ org_id: orgId, user_id: { $exists: false } }).toArray();
      orgDocs.forEach((doc) => {
        if (doc.provider === "github" || doc.provider === "bitbucket") {
          status[doc.provider] = {
            connected: true,
            username: doc.username,
            avatarUrl: doc.avatar_url,
            expiresAt: doc.expires_at,
          };
        }
      });

      // 2. Fetch personal connections (if userId provided) and override org-level ones
      if (userId) {
        const personalDocs = await collection.find({ user_id: userId, org_id: orgId }).toArray();
        personalDocs.forEach((doc) => {
          if (doc.provider === "github" || doc.provider === "bitbucket") {
            status[doc.provider] = {
              connected: true,
              username: doc.username,
              avatarUrl: doc.avatar_url,
              expiresAt: doc.expires_at,
            };
          }
        });

        // 3. Special case: If user is currently logged in via OAuth, ensure that provider is marked connected
        // even if no separate connection document exists (backward compatibility)
        const usersCollection = await getUsersCollection();
        const user = await usersCollection.findOne({ user_id: userId });
        if (user) {
          if (user.user_type === UserType.GITHUB && !status.github) {
            status.github = {
              connected: true,
              username: user.provider_username ?? "",
              avatarUrl: user.avatar_url ?? "",
              expiresAt: null,
            };
          } else if (user.user_type === UserType.BITBUCKET && !status.bitbucket) {
            status.bitbucket = {
              connected: true,
              username: user.provider_username ?? "",
              avatarUrl: user.avatar_url ?? "",
              expiresAt: null,
            };
          }
        }
      }

      return returnPayload(true, "Status retrieved", status);
    } catch (error: unknown) {
      logger.error("❌ Error in getStatus:", error);
      return returnPayload(false, "Failed to get status", { github: null, bitbucket: null });
    }
  }

  async disconnect(body: DisconnectRequest): Promise<ApiResponse<{ message: string }>> {
    const { org_id, provider } = body;
    try {
      // 1. Delete AWS secret
      const secretName = this.getSecretName(org_id, provider);
      await awsSecretsManagerService.deleteSecretByName(secretName);

      // 2. Delete MongoDB doc
      const collection = await getProviderConnectionsCollection();
      await collection.deleteOne({ org_id, provider });

      return returnPayload(true, "Disconnected successfully", { message: `Disconnected from ${provider}` });
    } catch (error: unknown) {
      logger.error("❌ Error in disconnect:", error);
      return returnPayload(false, "Failed to disconnect", { message: (error as Error).message });
    }
  }

  // async getRepositories(orgId: string, provider: ProviderType): Promise<ProviderRepo[]> {
  //   const collection = await getProviderConnectionsCollection();
  //   const connection = await collection.findOne({ org_id: orgId, provider });

  //   if (!connection) {
  //     throw new Error("No connection found");
  //   }

  //   // Check token expiration (applies to Bitbucket; GitHub OAuth tokens do not expire)
  //   let accessToken: string;
  //   if (connection.expires_at && new Date(connection.expires_at) < new Date()) {
  //     const refreshed = await this.refreshAccessToken(orgId, provider);
  //     if (!refreshed) {
  //       throw new TokenExpiredError("Token expired and could not be refreshed");
  //     }
  //     accessToken = refreshed;
  //   } else {
  //     // Fetch token from AWS SM (with retry when secret is transiently unavailable)
  //     const secretName = connection.aws_secret_name;
  //     let secretString: string | null = null;
  //     for (let attempt = 1; attempt <= 3; attempt++) {
  //       secretString = await awsSecretsManagerService.getSecretString(secretName);
  //       if (secretString) {
  //         break;
  //       }
  //       if (attempt < 3) {
  //         const delay = 200 * 2 ** (attempt - 1);
  //         logger.warn(`⚠️ Secret not found in getRepositories (attempt ${attempt}/3), retrying in ${delay}ms...`);
  //         await new Promise((resolve) => setTimeout(resolve, delay));
  //       }
  //     }
  //     if (!secretString) {
  //       throw new Error("Secret value empty");
  //     }
  //     const secret = JSON.parse(secretString) as OAuthTokenSecret;
  //     accessToken = secret.access_token;
  //   }

  //   let repos: ProviderRepo[] = [];

  //   if (provider === "github") {
  //     const res = await this.fetchWithRetry("https://api.github.com/user/repos?per_page=100&type=all", accessToken);

  //     if (res.status === 401 || res.status === 403) {
  //       await this.markRevoked(orgId, provider);
  //       throw new TokenExpiredError();
  //     }

  //     const data = (await res.json()) as Record<string, unknown>[];
  //     repos = data.map((r) => ({
  //       id: r.id as string | number,
  //       full_name: r.full_name as string,
  //       html_url: r.html_url as string,
  //       private: r.private as boolean,
  //       default_branch: r.default_branch as string,
  //       description: (r.description as string) || null,
  //     }));
  //   } else if (provider === "bitbucket") {
  //     // 1. Fetch Workspaces
  //     const workres = await this.fetchWithRetry("https://api.bitbucket.org/2.0/user/workspaces?pagelen=50", accessToken);

  //     if (workres.status === 401 || workres.status === 403) {
  //       await this.markRevoked(orgId, provider);
  //       throw new TokenExpiredError();
  //     }

  //     const workspaceData = (await workres.json()) as Record<string, unknown>;
  //     logger.info(`🔍 Bitbucket workspace API response: ${JSON.stringify(workspaceData)}`);

  //     const values = workspaceData.values as { slug: string }[] | undefined;
  //     const slugs = (values || []).map((w) => w.slug);
  //     logger.info(`🔍 Extracted workspace slugs: ${JSON.stringify(slugs)}`);

  //     if (slugs.length === 0) {
  //       return [];
  //     }

  //     // 2. Fetch Repos for each Workspace in parallel
  //     const repoPromises = slugs.map((slug) => ({
  //       slug,
  //       promise: this.fetchWithRetry(`https://api.bitbucket.org/2.0/repositories/${slug}?role=member&pagelen=100`, accessToken),
  //     }));

  //     const repoResponses = await Promise.all(repoPromises.map(async (p) => ({
  //       slug: p.slug,
  //       res: await p.promise,
  //     })));

  //     const allRepos: ProviderRepo[] = [];
  //     for (const { slug, res } of repoResponses) {
  //       if (!res.ok) {
  //         logger.error(`Failed to fetch repositories for workspace '${slug}': ${res.status} ${res.statusText}`);

  //         // For 404 errors, the workspace might exist but user has no access
  //         // Log and continue instead of throwing
  //         if (res.status === 404) {
  //           logger.warn(`⚠️ Skipping workspace '${slug}' - no access or workspace not found (404)`);
  //           continue;
  //         }

  //         // Try to get more error details for other errors
  //         let errorDetails = "";
  //         try {
  //           const errorData = await res.json();
  //           errorDetails = JSON.stringify(errorData);
  //         } catch {
  //           // Ignore JSON parse errors
  //         }

  //         throw new WorkspaceFetchError(
  //           `Workspace repository fetch failed for workspace '${slug}' with status ${res.status}${errorDetails ? `. Details: ${errorDetails}` : ""}`,
  //         );
  //       }

  //       const data = (await res.json()) as { values: Record<string, unknown>[] };
  //       const mapped = (data.values || []).map((r) => {
  //         const links = r.links as { html: { href: string } };
  //         const mainbranch = r.mainbranch as { name: string } | undefined;
  //         return {
  //           id: r.uuid as string,
  //           full_name: r.full_name as string,
  //           html_url: links.html.href,
  //           private: r.is_private as boolean,
  //           default_branch: mainbranch?.name || "master",
  //           description: (r.description as string) || null,
  //         };
  //       });
  //       allRepos.push(...mapped);
  //     }
  //     repos = allRepos;
  //   }

  //   return repos;
  // }
  async getRepositories(orgId: string, provider: ProviderType, userId?: string): Promise<ProviderRepo[]> {
    const accessToken = await this.getValidAccessToken(orgId, provider, userId);
    if (!accessToken) {
      throw new Error("No connection found or token could not be retrieved");
    }

    let repos: ProviderRepo[] = [];

    if (provider === "github") {
      const res = await this.fetchWithRetry("https://api.github.com/user/repos?per_page=100&type=all", accessToken);
      if (res.status === 401 || res.status === 403) {
        await this.markRevoked(orgId, provider, userId);
        throw new TokenExpiredError();
      }
      const data = (await res.json()) as Record<string, any>[];
      repos = data.map((r) => ({
        id: r.id,
        full_name: r.full_name,
        html_url: r.html_url,
        private: r.private,
        default_branch: r.default_branch,
        description: r.description || null,
      }));
    } else if (provider === "bitbucket") {
      const allRepos: ProviderRepo[] = [];
      let workspaceUrl: string | null = "https://api.bitbucket.org/2.0/user/workspaces?pagelen=50";

      while (workspaceUrl) {
        const workres = await this.fetchWithRetry(workspaceUrl, accessToken);

        if (!workres.ok) {
          if (workres.status === 401 || workres.status === 403) {
            await this.markRevoked(orgId, provider, userId);
            throw new TokenExpiredError();
          }
          throw new WorkspaceFetchError(`Workspace fetch failed: ${workres.status}`);
        }

        const workspaceData = await workres.json();
        const workspaceValues = workspaceData.values || [];

        for (const v of workspaceValues) {
          // Updated extraction: check both direct slug and nested workspace.slug
          // to handle inconsistent API behavior across different workspace types
          const slug = v.slug || v.workspace?.slug;

          if (!slug) {
            logger.warn(`⚠️ Skipping a workspace entry because slug was undefined: ${JSON.stringify(v)}`);
            continue;
          }

          let repoUrl: string | null = `https://api.bitbucket.org/2.0/repositories/${slug}?role=member&pagelen=100`;

          while (repoUrl) {
            const repoRes = await this.fetchWithRetry(repoUrl, accessToken);

            if (!repoRes.ok) {
              logger.error(`❌ Failed repos for workspace '${slug}': ${repoRes.status}`);
              break;
            }

            const repoData = await repoRes.json();
            const pageRepos = (repoData.values || []).map((r: any) => ({
              id: r.uuid,
              full_name: r.full_name,
              html_url: r.links.html.href,
              private: r.is_private,
              default_branch: r.mainbranch?.name || "master",
              description: r.description || null,
            }));

            allRepos.push(...pageRepos);
            repoUrl = repoData.next || null;
          }
        }
        workspaceUrl = workspaceData.next || null;
      }
      repos = allRepos;
    }

    return repos;
  }
  async getAccessToken(orgId: string, provider: ProviderType): Promise<string | null> {
    return this.getValidAccessToken(orgId, provider);
  }
}

// Singleton
let instance: ProviderService | null = null;
export function getProviderService(): ProviderService {
  if (!instance) {
    instance = new ProviderService();
  }
  return instance;
}

/**
 * Global accessor for repository listing
 */
export async function getRepositories(orgId: string, provider: ProviderType, userId?: string): Promise<ProviderRepo[]> {
  return getProviderService().getRepositories(orgId, provider, userId);
}

/**
 * Global accessor for OAuth tokens
 */
export async function getOAuthToken(orgId: string, provider: ProviderType): Promise<string | null> {
  return getProviderService().getAccessToken(orgId, provider);
}
