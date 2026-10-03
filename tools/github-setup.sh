#!/usr/bin/env bash
# Apply this repository's GitHub settings: merges, Actions, labels, and,
# once the repository is public, the ruleset on main and the security
# features GitHub's free plan offers only to public repositories.
# Idempotent: run it again after any change, and once right after the
# repository is made public.
#
#   tools/github-setup.sh                 # Trainnr-AI/trainnr
#   REPO=owner/name tools/github-setup.sh
#
# Who can merge: only an account with write access to the repository can
# merge a pull request, and the ruleset below makes even those wait for a
# code owner's approval and green checks. The organisation's admins are
# the only bypass. The script ends by listing every account that has
# write access, so the list can be checked by eye.
set -euo pipefail
REPO="${REPO:-Trainnr-AI/trainnr}"
OWNER="${REPO%%/*}"
api() { gh api -H "Accept: application/vnd.github+json" "$@"; }
say() { printf '%s\n' "$*"; }

PRIVATE=$(api "repos/$REPO" --jq .private)
say "repository: $REPO (private: $PRIVATE)"

# --- merging -----------------------------------------------------------
# The homepage points at the repository until trainnr.ai serves its own
# page over HTTPS (it is a parked domain as of 2026-10-03); the
# organisation's profile is left as it is.
api -X PATCH "repos/$REPO" \
  -F has_issues=true -F has_discussions=true -F has_wiki=false -F has_projects=false \
  -F allow_forking=true \
  -f homepage=https://github.com/Trainnr-AI/trainnr \
  -F allow_merge_commit=false -F allow_squash_merge=true -F allow_rebase_merge=true \
  -F allow_auto_merge=false -F delete_branch_on_merge=true -F allow_update_branch=true \
  -f squash_merge_commit_title=PR_TITLE -f squash_merge_commit_message=PR_BODY >/dev/null
say "merging: squash or rebase only, branches deleted after merge (web commit sign-off is enforced by the organisation)"

# --- Actions -----------------------------------------------------------
# The default token reads only; a workflow asks for more by name (the
# badge job, the Scorecard upload). Actions may never approve a pull
# request. Only GitHub's own, verified creators' and the pinned actions
# below may run.
api -X PUT "repos/$REPO/actions/permissions/workflow" \
  -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false >/dev/null
api -X PUT "repos/$REPO/actions/permissions" -F enabled=true -f allowed_actions=selected >/dev/null
api -X PUT "repos/$REPO/actions/permissions/selected-actions" --input - >/dev/null <<'JSON'
{"github_owned_allowed": true, "verified_allowed": true,
 "patterns_allowed": ["astral-sh/setup-uv@*", "dtolnay/rust-toolchain@*", "Swatinem/rust-cache@*", "ossf/scorecard-action@*"]}
JSON
say "actions: read-only token, no PR approvals by Actions, allow-listed actions only"
# A first-time or outside contributor's workflow waits for a maintainer's
# approval before it runs (public repositories only).
if [ "$PRIVATE" = "false" ]; then
  api -X PUT "repos/$REPO/actions/permissions/fork-pr-contributor-approval" \
    -f approval_policy=all_external_contributors >/dev/null
  say "actions: every outside contributor's run waits for approval"
fi

# --- labels ------------------------------------------------------------
uri() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }
label() {  # name colour description
  if api "repos/$REPO/labels/$(uri "$1")" >/dev/null 2>&1; then
    api -X PATCH "repos/$REPO/labels/$(uri "$1")" -f color="$2" -f description="$3" >/dev/null
  else
    api -X POST "repos/$REPO/labels" -f name="$1" -f color="$2" -f description="$3" >/dev/null
  fi
}
label "needs-triage"   "fbca04" "New; a maintainer has not looked yet"
label "bug"            "d73a4a" "Something does not do what it says"
label "enhancement"    "a2eeef" "A workflow trainnr should support"
label "robot-support"  "0e8a16" "A robot, actuator, telemetry format or simulator"
label "documentation"  "0075ca" "Docs only"
label "good first issue" "7057ff" "Small, well-scoped, a good start"
label "help wanted"    "008672" "A maintainer would welcome a pull request"
label "dependencies"   "0366d6" "A dependency update (Dependabot)"
label "conduct"        "b60205" "A Code of Conduct report"
label "area: studio"   "c5def5" "The desktop app"
label "area: mcp"      "c5def5" "The MCP server and its tools"
label "area: identify" "c5def5" "System identification and drift"
label "area: train"    "c5def5" "Training and evaluation, trainnr-mjlab"
label "area: deploy"   "c5def5" "Export, gate, pre-flight"
label "area: data"     "c5def5" "Demos, datasets, recordings, scenes"
say "labels: triage, kinds and areas"

# --- public-only: protection and security --------------------------------
if [ "$PRIVATE" = "true" ]; then
  say "skipped while private (GitHub's free plan): the main ruleset, secret scanning and push protection, private vulnerability reporting, Dependabot alerts and fixes, CodeQL. Run this script again after the repository is public."
else
  # main: no direct pushes, no force pushes, no deletion; a pull request
  # with a code owner's approval (CODEOWNERS names the maintainer), the
  # latest push approved, conversations resolved, and the gates green.
  # Organisation admins may bypass, so the maintainer can still merge
  # their own pull requests.
  # With one maintainer, the approval rule cannot be met on the
  # maintainer's own pull requests (nobody else can approve), so those
  # merge through the admin bypass; for anyone else's it holds. Only
  # always-running jobs are required: path-filtered workflows
  # (studio-platforms, mjlab) and the python-matrix and macos jobs are
  # not, until they have run green for a while.
  RULES=$(cat <<'JSON'
{"name": "main", "target": "branch", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
 "bypass_actors": [{"actor_id": 1, "actor_type": "OrganizationAdmin", "bypass_mode": "always"}],
 "rules": [
  {"type": "deletion"}, {"type": "non_fast_forward"}, {"type": "required_linear_history"},
  {"type": "pull_request", "parameters": {
     "required_approving_review_count": 1, "require_code_owner_review": true,
     "dismiss_stale_reviews_on_push": true, "require_last_push_approval": true,
     "required_review_thread_resolution": true, "allowed_merge_methods": ["squash", "rebase"]}},
  {"type": "required_status_checks", "parameters": {
     "strict_required_status_checks_policy": true,
     "required_status_checks": [{"context": "fast-gates"}, {"context": "sign-off"}, {"context": "supply-chain"}, {"context": "package"}]}}
 ]}
JSON
)
  TAGS=$(cat <<'JSON'
{"name": "release tags", "target": "tag", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["refs/tags/v*"], "exclude": []}},
 "bypass_actors": [{"actor_id": 1, "actor_type": "OrganizationAdmin", "bypass_mode": "always"}],
 "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}]}
JSON
)
  for body in "$RULES" "$TAGS"; do
    name=$(printf %s "$body" | python3 -c 'import json, sys; print(json.load(sys.stdin)["name"])')
    id=$(api "repos/$REPO/rulesets" --jq ".[] | select(.name==\"$name\") | .id" | head -1)
    if [ -n "$id" ]; then
      printf %s "$body" | api -X PUT "repos/$REPO/rulesets/$id" --input - >/dev/null
    else
      printf %s "$body" | api -X POST "repos/$REPO/rulesets" --input - >/dev/null
    fi
    say "ruleset: $name"
  done
  api -X PATCH "repos/$REPO" --input - >/dev/null <<'JSON'
{"security_and_analysis": {"secret_scanning": {"status": "enabled"},
  "secret_scanning_push_protection": {"status": "enabled"}}}
JSON
  api -X PUT "repos/$REPO/private-vulnerability-reporting" >/dev/null
  api -X PUT "repos/$REPO/vulnerability-alerts" >/dev/null
  api -X PUT "repos/$REPO/automated-security-fixes" >/dev/null
  api -X PATCH "repos/$REPO/code-scanning/default-setup" -f state=configured -f query_suite=default >/dev/null || \
    say "code scanning: enable CodeQL's default setup by hand (Settings, Code security)"
  say "security: secret scanning with push protection, private vulnerability reporting, Dependabot alerts and fixes, CodeQL"
fi

# --- who can merge ------------------------------------------------------
say "accounts with write access (only these can merge):"
api "repos/$REPO/collaborators?affiliation=all&per_page=100" \
  --jq '.[] | select(.permissions.push or .permissions.admin or .permissions.maintain) | "  \(.login) (\(.role_name))"'
DEFAULT=$(api "orgs/$OWNER" --jq .default_repository_permission 2>/dev/null || echo unknown)
TWOFA=$(api "orgs/$OWNER" --jq .two_factor_requirement_enabled 2>/dev/null || echo unknown)
say "organisation: members' default permission = $DEFAULT (read keeps merging to the list above); two-factor required = $TWOFA"
