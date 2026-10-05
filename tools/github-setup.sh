#!/usr/bin/env bash
# Apply this repository's GitHub settings: merges, Actions, labels, and,
# once the repository is public, the ruleset on main and the security
# features GitHub's free plan offers only to public repositories.
# Idempotent: run it again after any change, and once right after the
# repository is made public. It ends by reading the settings back and
# exits non-zero, naming each one, when any did not take.
#
#   tools/github-setup.sh                 # Trainnr-AI/trainnr
#   REPO=owner/name tools/github-setup.sh
#
# Who can merge: only an account with write access to the repository can
# merge a pull request, and the ruleset below makes even those wait for a
# code owner's approval and green checks. The organisation's admins are
# the only bypass, and only through a pull request (never a direct push).
# The script lists every account that has write access, so the list can
# be checked by eye.
set -euo pipefail
REPO="${REPO:-Trainnr-AI/trainnr}"
OWNER="${REPO%%/*}"
api() { gh api -H "Accept: application/vnd.github+json" "$@"; }
say() { printf '%s\n' "$*"; }
# Every setting is applied this way: a refused call is named and the
# script goes on, so one refusal never leaves main without its ruleset;
# the read-back at the end decides the exit code (review, 2026-10-05).
apply() {  # description, then the api arguments
  local what=$1; shift
  api "$@" >/dev/null || say "  NOT applied: $what (named again in the read-back)"
}

PRIVATE=$(api "repos/$REPO" --jq .private)
say "repository: $REPO (private: $PRIVATE)"

# --- merging -----------------------------------------------------------
# The homepage points at the repository until trainnr.ai serves its own
# page over HTTPS (it is a parked domain as of 2026-10-03); the
# organisation's profile is left as it is.
apply "merge settings" -X PATCH "repos/$REPO" \
  -F has_issues=true -F has_discussions=true -F has_wiki=false -F has_projects=false \
  -f homepage=https://github.com/Trainnr-AI/trainnr \
  -F allow_merge_commit=false -F allow_squash_merge=true -F allow_rebase_merge=true \
  -F allow_auto_merge=false -F delete_branch_on_merge=true -F allow_update_branch=true \
  -f squash_merge_commit_title=PR_TITLE -f squash_merge_commit_message=COMMIT_MESSAGES
# A squash commit's message is the pull request's commit messages, so each
# commit's Signed-off-by line (the DCO) reaches main; PR_BODY dropped them.
say "merging: squash or rebase only, a squash keeps every commit's sign-off, branches deleted after merge (web commit sign-off is enforced by the organisation)"

# --- Actions -----------------------------------------------------------
# The default token reads only; a workflow asks for more by name (the
# badge job, the Scorecard upload, the release). Actions may never approve
# a pull request. Only GitHub's own actions and the four named below may
# run (a "verified creator" badge is not a review of this repository's
# needs, so it grants nothing), and every action must be pinned to a full
# commit SHA, which the workflows already do.
apply "the workflow token" -X PUT "repos/$REPO/actions/permissions/workflow" \
  -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false
apply "the allowed actions" -X PUT "repos/$REPO/actions/permissions" \
  -F enabled=true -f allowed_actions=selected -F sha_pinning_required=true
apply "the named actions" -X PUT "repos/$REPO/actions/permissions/selected-actions" --input - <<'JSON'
{"github_owned_allowed": true, "verified_allowed": false,
 "patterns_allowed": ["astral-sh/setup-uv@*", "dtolnay/rust-toolchain@*", "Swatinem/rust-cache@*", "ossf/scorecard-action@*"]}
JSON
say "actions: read-only token, no PR approvals by Actions, GitHub's own and four named actions only, each pinned to a commit SHA"
# A first-time or outside contributor's workflow waits for a maintainer's
# approval before it runs (public repositories only).
if [ "$PRIVATE" = "false" ]; then
  # Forks are how outside contributors open pull requests; GitHub allows
  # them on a private repository only when the organisation does.
  # Neither stops the script: a refusal here must not leave main without
  # the rulesets below (the read-back at the end names what did not take).
  apply "forks allowed" -X PATCH "repos/$REPO" -F allow_forking=true
  apply "outside contributors' runs wait for approval" \
    -X PUT "repos/$REPO/actions/permissions/fork-pr-contributor-approval" \
    -f approval_policy=all_external_contributors
  say "forks: allowed; every outside contributor's run waits for approval"
fi

# --- labels ------------------------------------------------------------
uri() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }
label() {  # name colour description
  if api "repos/$REPO/labels/$(uri "$1")" >/dev/null 2>&1; then
    apply "label $1" -X PATCH "repos/$REPO/labels/$(uri "$1")" -f color="$2" -f description="$3"
  else
    apply "label $1" -X POST "repos/$REPO/labels" -f name="$1" -f color="$2" -f description="$3"
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
  # Organisation admins may bypass, and only through a pull request
  # (bypass_mode "pull_request"): with one maintainer, the approval rule
  # cannot be met on the maintainer's own pull requests (nobody else can
  # approve), so those merge through the bypass; for anyone else's it
  # holds, and nobody pushes to main directly. Only always-running jobs
  # are required: path-filtered workflows (studio-platforms, mjlab), the
  # python-matrix and macos jobs, the advisory audit (a moving database)
  # and the pull request title check are not. Each required check is
  # pinned to the GitHub Actions app (integration 15368), so a status
  # posted by anything else under the same name does not satisfy it.
  RULES=$(cat <<'JSON'
{"name": "main", "target": "branch", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
 "bypass_actors": [{"actor_id": 1, "actor_type": "OrganizationAdmin", "bypass_mode": "pull_request"}],
 "rules": [
  {"type": "deletion"}, {"type": "non_fast_forward"}, {"type": "required_linear_history"},
  {"type": "pull_request", "parameters": {
     "required_approving_review_count": 1, "require_code_owner_review": true,
     "dismiss_stale_reviews_on_push": true, "require_last_push_approval": true,
     "required_review_thread_resolution": true, "allowed_merge_methods": ["squash", "rebase"]}},
  {"type": "required_status_checks", "parameters": {
     "strict_required_status_checks_policy": true,
     "required_status_checks": [
       {"context": "fast-gates", "integration_id": 15368}, {"context": "sign-off", "integration_id": 15368},
       {"context": "supply-chain", "integration_id": 15368}, {"context": "package", "integration_id": 15368}]}}
 ]}
JSON
)
  # Release tags (v*) and the artifacts-* tags the large-file archives
  # hang from: only the organisation's admins create, move or delete
  # them.
  TAGS=$(cat <<'JSON'
{"name": "release tags", "target": "tag", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["refs/tags/v*", "refs/tags/artifacts-*"], "exclude": []}},
 "bypass_actors": [{"actor_id": 1, "actor_type": "OrganizationAdmin", "bypass_mode": "always"}],
 "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}]}
JSON
)
  for body in "$RULES" "$TAGS"; do
    name=$(printf %s "$body" | python3 -c 'import json, sys; print(json.load(sys.stdin)["name"])')
    id=$(api "repos/$REPO/rulesets" --jq ".[] | select(.name==\"$name\") | .id" | head -1) || id=""
    if [ -n "$id" ]; then
      printf %s "$body" | apply "ruleset $name" -X PUT "repos/$REPO/rulesets/$id" --input -
    else
      printf %s "$body" | apply "ruleset $name" -X POST "repos/$REPO/rulesets" --input -
    fi
    say "ruleset: $name"
  done
  apply "secret scanning" -X PATCH "repos/$REPO" --input - <<'JSON'
{"security_and_analysis": {"secret_scanning": {"status": "enabled"},
  "secret_scanning_push_protection": {"status": "enabled"}}}
JSON
  apply "private vulnerability reporting" -X PUT "repos/$REPO/private-vulnerability-reporting"
  apply "Dependabot alerts" -X PUT "repos/$REPO/vulnerability-alerts"
  apply "Dependabot fixes" -X PUT "repos/$REPO/automated-security-fixes"
  api -X PATCH "repos/$REPO/code-scanning/default-setup" -f state=configured -f query_suite=default >/dev/null || \
    say "code scanning: enable CodeQL's default setup by hand (Settings, Code security)"
  # A published release's tag and assets can no longer change; the
  # release workflow attaches the archives to a draft, which stays
  # editable until a maintainer publishes it.
  apply "immutable releases" -X PUT "repos/$REPO/immutable-releases"
  say "security: secret scanning with push protection, private vulnerability reporting, Dependabot alerts and fixes, CodeQL, immutable releases"
fi

# --- read back -------------------------------------------------------------
# Every setting above, read back from the API: a call that was accepted
# but did not take (a plan limit, an organisation policy) is named here,
# and the script exits non-zero.
missing=()
need() {  # description, then a command that succeeds when the setting holds
  local what=$1; shift
  if "$@" >/dev/null 2>&1; then say "  ok: $what"; else missing+=("$what"); say "  MISSING: $what"; fi
}
is() {  # expected value, then the api arguments whose output must equal it
  local want=$1; shift
  [ "$(api "$@" 2>/dev/null)" = "$want" ]
}
say "reading the settings back:"
need "squash commits keep the commit messages" is COMMIT_MESSAGES "repos/$REPO" --jq .squash_merge_commit_message
need "workflow token read-only" is read "repos/$REPO/actions/permissions/workflow" --jq .default_workflow_permissions
need "selected actions only" is selected "repos/$REPO/actions/permissions" --jq .allowed_actions
need "actions pinned to a full commit SHA" is true "repos/$REPO/actions/permissions" --jq .sha_pinning_required
need "verified creators' actions not allowed" is false "repos/$REPO/actions/permissions/selected-actions" --jq .verified_allowed
if [ "$PRIVATE" = "false" ]; then
  RULESETS=$(api "repos/$REPO/rulesets" --jq '.[] | select(.enforcement == "active") | .name' 2>/dev/null || true)
  need "ruleset on main (active)" grep -qx 'main' <<<"$RULESETS"
  need "ruleset on release tags (active)" grep -qx 'release tags' <<<"$RULESETS"
  need "forks allowed" is true "repos/$REPO" --jq .allow_forking
  need "private vulnerability reporting" is true "repos/$REPO/private-vulnerability-reporting" --jq .enabled
  need "secret scanning" is enabled "repos/$REPO" --jq .security_and_analysis.secret_scanning.status
  need "secret scanning push protection" is enabled "repos/$REPO" --jq .security_and_analysis.secret_scanning_push_protection.status
  need "Dependabot alerts (vulnerability-alerts 204)" api "repos/$REPO/vulnerability-alerts"
  need "outside contributors' runs wait for approval" is all_external_contributors \
    "repos/$REPO/actions/permissions/fork-pr-contributor-approval" --jq .approval_policy
  need "immutable releases" is true "repos/$REPO/immutable-releases" --jq .enabled
else
  say "  (the public-only settings are read back once the repository is public)"
fi

# --- who can merge ------------------------------------------------------
say "accounts with write access (only these can merge):"
api "repos/$REPO/collaborators?affiliation=all&per_page=100" \
  --jq '.[] | select(.permissions.push or .permissions.admin or .permissions.maintain) | "  \(.login) (\(.role_name))"'
DEFAULT=$(api "orgs/$OWNER" --jq .default_repository_permission 2>/dev/null || echo unknown)
TWOFA=$(api "orgs/$OWNER" --jq .two_factor_requirement_enabled 2>/dev/null || echo unknown)
say "organisation: members' default permission = $DEFAULT (read keeps merging to the list above); two-factor required = $TWOFA"

if [ "${#missing[@]}" -gt 0 ]; then
  say "NOT APPLIED (${#missing[@]}):"
  printf '  - %s\n' "${missing[@]}"
  exit 1
fi
say "every setting read back as applied"
