# Pre-Deploy Checklist

Run this before every `git push origin main` (production deploy).

---

## Step 1 — Automated Tests

```bash
bash run_tests.sh
```

All tests must pass before continuing. If any fail → fix before deploying.

---

## Step 2 — Manual Smoke Tests (5 minutes)

Open the **staging URL** and run through each item below.
Mark ✅ when OK, ❌ when broken.

### AUTH
- [ ] Login with valid credentials → lands on Orders page
- [ ] Login with wrong password → shows error, no crash
- [ ] Logout → clears session, redirects to login
- [ ] Reload page after login → session restored (URL token works)
- [ ] Open app in incognito → shows login form

### NAVIGATION
- [ ] Bottom tabs: New / Borrador / Orders / Receive all load without error
- [ ] Top bar icons: Catalog (🧾), History (📈), Manage Org (⚙️) all load
- [ ] Logout link in top bar works

### NEW ORDER (Voice + Text)
- [ ] Type text in composer → parses items correctly
- [ ] Parsed items appear in the list
- [ ] Strike through an item → disappears from totals
- [ ] Click "Add to Draft" → draft is created
- [ ] App navigates to Orders page after save
- [ ] Loading spinner shows during catalog load

### VOICE INPUT
- [ ] Mic button opens audio overlay
- [ ] Record 3–5 seconds → transcript appears
- [ ] Transcript is parsed into items (with quantities)
- [ ] If ambiguous item → suggestion selector appears
- [ ] Picking a suggestion → item confirmed

### ORDERS PAGE (Pedidos / Listo)
- [ ] Loading spinner shows while fetching orders
- [ ] Order list appears after load
- [ ] Select an order → order detail renders
- [ ] Status chip shows correct emoji (📝 Borrador / 📤 Listo)

### BORRADOR PAGE
- [ ] Loading spinner shows
- [ ] Draft orders appear in dropdown
- [ ] Select a draft → line items render
- [ ] Line editor works (can change quantity)

### RECEIVE / DASHBOARD
- [ ] Loading spinner shows while loading dashboard
- [ ] "Pending" tab loads with pending receive items
- [ ] "Open incidences" tab loads
- [ ] Re-deliveries tab loads
- [ ] Credit notes tab loads
- [ ] Clicking a provider panel → receive form opens

### CATALOG
- [ ] Loading spinner shows
- [ ] Product list renders
- [ ] Search / filter by provider works
- [ ] Edit a product → saves correctly
- [ ] Add new product → appears in list

### HISTORY
- [ ] Loading spinner shows
- [ ] History tab loads with closed orders
- [ ] Reports tab loads

---

## Step 3 — Database Check

Quick sanity check that no stale data or migration is needed:

- [ ] A test order created in staging is visible in the **staging** Supabase dashboard
- [ ] No errors in Supabase logs (Supabase → Logs → Postgres)
- [ ] Production Supabase has no unexpected schema changes

---

## Step 4 — Sentry Check

- [ ] Go to sentry.io → check for any **new unresolved errors** from staging
- [ ] No critical errors introduced by this deploy

---

## Step 5 — Final Deploy

Only if all above pass:

```bash
git checkout main
git merge staging
git push origin main
```

Then **watch Render deploy logs** for 3 minutes:
- [ ] Build completes without error
- [ ] Health check passes (app loads)
- [ ] Do a quick login on production URL

---

## Rollback Procedure

If something breaks in production after deploy:

**Option A — Render (30 seconds):**
```
Render Dashboard → voicecount-prod → Deploys → click previous deploy → "Rollback"
```

**Option B — Git:**
```bash
git checkout main
git revert HEAD --no-edit
git push origin main
```

---

## Notes

- Never push directly to `main` — always go through `staging` first
- Never test with real client data on staging — use your own test account
- If a test fails and you're unsure why → check Sentry before deploying
