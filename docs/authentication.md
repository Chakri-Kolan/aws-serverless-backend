# Authentication and token lifecycle

## Request flow

1. A client signs in against the deployed Cognito user pool and public app client.
2. Cognito returns ID, access, and refresh tokens.
3. The client sends the **ID token** in the `Authorization` header for `/items` requests.
4. API Gateway validates the token signature, expiry, issuer, and user-pool membership.
5. Lambda receives validated claims, requires `sub`, and scopes every DynamoDB operation to it.

The REST API methods currently have no OAuth scopes, so the Cognito authorizer
treats the supplied token as an ID token. `/health` is intentionally public.

## Credential and token storage

- The app client is public and has no client secret. A browser or mobile app must never embed AWS credentials.
- Prefer a backend-for-frontend that stores refresh tokens in `Secure`, `HttpOnly`, `SameSite` cookies.
- If a pure browser client is necessary, keep access and ID tokens in memory. Do not persist tokens in `localStorage`.
- Native applications should use the operating system's secure credential store.
- Never log tokens, authorization headers, passwords, authorization codes, or refresh tokens.
- Developers should authenticate to AWS with IAM Identity Center/SSO or another short-lived `credential_process`; do not put long-lived access keys in `.env` files.
- CI only packages the stack. A future deployment workflow should use GitHub OIDC and an environment-scoped, least-privilege AWS role instead of repository secrets containing AWS keys.

## Token lifecycle

- ID and access tokens expire after one hour.
- Refresh tokens expire after 30 days.
- Refresh before the ID token expires and replace the in-memory ID token atomically.
- On logout, discard local tokens and call Cognito revocation/global sign-out where the client supports it.
- A `401` means the token is missing, expired, revoked, or invalid. Reauthenticate; do not retry the same token indefinitely.
- Rotate/revoke sessions after suspected compromise and review Cognito authentication events.

## Obtain a token for manual testing

After deployment, create and confirm a non-production test user using an approved
admin process. Then use an SRP-capable Cognito client or AWS Amplify to sign in.
The app client enables `USER_SRP_AUTH`; this project deliberately does not enable
the password-in-CLI auth flow or store test-user passwords.

Use the resulting ID token without writing it to shell history:

```bash
read -rsp "Cognito ID token: " ID_TOKEN && echo
curl --fail-with-body \
  -H "Authorization: $ID_TOKEN" \
  "$API_URL/items"
unset ID_TOKEN
```
