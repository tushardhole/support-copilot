# Troubleshooting Common Issues

## Login Problems

### Forgot Password
Click **Forgot Password** on the login page. Enter your email.
You'll receive a reset link valid for 1 hour.
If you don't receive it within 5 minutes, check your spam folder.

### Account Locked
After 5 failed login attempts, your account is locked for 15 minutes.
Contact support if you need an immediate unlock.

### SSO / SAML Issues
Ensure your IT admin has configured the ACS URL:
`https://app.acme.io/auth/saml/callback`
Entity ID: `https://app.acme.io`

## Performance Issues

### Slow Dashboard
- Clear browser cache (Ctrl+Shift+R / Cmd+Shift+R).
- Disable browser extensions one by one to identify conflicts.
- Try an incognito/private window.
- Check [status.acme.io](https://status.acme.io) for ongoing incidents.

### File Uploads Failing
- Maximum file size: 250 MB per file.
- Supported formats: PDF, DOCX, XLSX, PNG, JPG, CSV.
- If upload hangs > 30s, check your network and retry.

## Data & Exports

### Export to CSV
Go to the project → **⋮ Menu** → **Export → CSV**.
Exports include all tasks, assignees, due dates, and custom fields.

### Data Retention
Data is retained for 90 days after account cancellation.
Request a full export before cancelling to preserve your data.

## Integrations

### Slack Integration
1. Go to **Settings → Integrations → Slack**.
2. Click **Connect** and authorize in Slack.
3. Choose which projects post notifications to which channels.

### Webhook Setup
Webhooks fire on task create/update/delete events.
Configure endpoint URL at **Settings → Integrations → Webhooks**.
We send a `X-Acme-Signature` header (HMAC-SHA256) for verification.
