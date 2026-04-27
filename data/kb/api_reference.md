# API Reference

## Authentication

All API requests require a Bearer token in the Authorization header:

```
Authorization: Bearer <your_api_token>
```

Generate tokens at **Settings → API → New Token**.
Tokens do not expire but can be revoked at any time.

## Base URL

```
https://api.acme.io/v1
```

## Rate Limits

| Plan | Requests/minute | Requests/day |
|------|----------------|--------------|
| Starter | 60 | 10,000 |
| Pro | 300 | 100,000 |
| Business | 1,000 | Unlimited |

Rate limit headers returned on every response:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset` (Unix timestamp)

## Endpoints

### List Projects
```
GET /projects
```
Returns all projects the token owner can access.

### Get Project
```
GET /projects/{project_id}
```

### List Tasks
```
GET /projects/{project_id}/tasks
```
Query params: `status` (open|done), `assignee_id`, `page`, `per_page` (max 100).

### Create Task
```
POST /projects/{project_id}/tasks
```
Body:
```json
{
  "title": "Fix login bug",
  "description": "Users can't log in with SSO",
  "assignee_id": "usr_123",
  "due_date": "2025-06-01",
  "priority": "high"
}
```

### Update Task
```
PATCH /tasks/{task_id}
```

### Orders

```
GET /orders/{order_id}
```
Returns order status, items, shipping info, and invoice URL.

```
POST /orders/{order_id}/refund
```
Body: `{ "reason": "damaged", "items": ["item_abc"] }`

## Webhooks

Events: `task.created`, `task.updated`, `task.deleted`, `order.shipped`, `order.refunded`.

Retry policy: 3 attempts with exponential back-off (1s, 5s, 25s).
