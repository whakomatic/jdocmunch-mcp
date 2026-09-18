#!/usr/bin/env bash
# Keep this fork's work branches current with jgravelle/jdocmunch-mcp.
#
#   ./scripts/fork-sync.sh sync [branch]     fast-forward master, rebase branch onto it
#   ./scripts/fork-sync.sh verify [branch]   check the rebase did not silently lose our work
#   ./scripts/fork-sync.sh status            where every branch sits relative to upstream
#
# `branch` defaults to the branch you are on. `sync` records the pre-rebase tip
# in refs/fork-sync/<branch>/pre-rebase and runs `verify` against it once the
# rebase lands. After resolving conflicts by hand, finish with
# `git rebase --continue`, then re-run `./scripts/fork-sync.sh verify <branch>` --
# the recorded ref is still there.
#
# The layout this assumes, and enforces:
#
#   upstream/master   jgravelle's tree, the only base that matters
#   master            a PRISTINE MIRROR of upstream/master; carries no local work
#   <work branches>   each sitting directly on upstream/master, so each is PR-able
#
# master holding even one local commit is what this script exists to prevent: it
# turns every future sync from a fast-forward into a merge, and it makes a work
# branch's diff against upstream contain commits the PR never meant to carry.
# Local work belongs on a branch, never on master.
#
# NOTE ON `git stash`: a PreToolUse hook in this environment refuses stash,
# reset, restore and `checkout -- <path>`, because several agent sessions share
# one checkout and those commands discard a peer's uncommitted work. So this
# script never stashes; it refuses a dirty tree and tells you to commit.

set -euo pipefail

# Anchor every git call at the repo this script lives in, not at the caller's
# cwd. Run from elsewhere and git resolves whatever repository happens to be
# above you, or none, and the first failure is a misleading "No 'upstream'
# remote" for a tree that was never this fork.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

UPSTREAM_REMOTE=upstream
UPSTREAM=upstream/master
MIRROR=master

die() { printf '%s\n' "$*" >&2; exit 1; }

# The divergence surface: files where a branch differs from its base.
# A file that drops off this list between two runs was either absorbed
# upstream (fine) or silently lost by the rebase (not fine).
divergent_files() {
    git diff --name-only "$1" "$2"
}

resolve_branch() {
    local b="${1:-}"
    if [ -z "$b" ]; then
        b=$(git symbolic-ref --quiet --short HEAD) \
            || die "Detached HEAD, and no branch given. usage: $0 $2 <branch>"
    fi
    [ "$b" != "$MIRROR" ] \
        || die "$MIRROR is the upstream mirror and carries no work of its own. Name a work branch."
    git rev-parse --verify --quiet "$b" >/dev/null || die "No such branch: $b"
    printf '%s' "$b"
}

require_upstream_remote() {
    git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1 || die \
"No '$UPSTREAM_REMOTE' remote. Add the source repo this fork tracks:

    git remote add $UPSTREAM_REMOTE https://github.com/jgravelle/jdocmunch-mcp.git"
}

# master must contain nothing upstream does not. Checked BEFORE the fetch moves
# upstream, so the message can distinguish "behind" (normal) from "carries local
# commits" (the thing that breaks every future sync).
assert_mirror_clean() {
    local ahead
    ahead=$(git rev-list --count "$UPSTREAM..$MIRROR" 2>/dev/null || echo 0)
    [ "$ahead" = "0" ] || die \
"$MIRROR is $ahead commit(s) ahead of $UPSTREAM, so it is not a mirror and
cannot fast-forward. Those commits belong on a work branch:

    git branch -f my-work $MIRROR
    git branch -f $MIRROR $UPSTREAM

The first keeps them, the second makes $MIRROR a mirror again. Then re-run
this; 'fork-sync.sh status' shows where everything sits.

(Note that this rewrites $MIRROR, so a $MIRROR already pushed to your own
fork will need --force-with-lease on its next push.)"
}

cmd_status() {
    require_upstream_remote
    echo "==> Fetching $UPSTREAM_REMOTE"
    git fetch "$UPSTREAM_REMOTE" --quiet
    printf '\n%-34s %s\n' "$UPSTREAM" "$(git rev-parse --short "$UPSTREAM")"
    local ahead behind
    ahead=$(git rev-list --count "$UPSTREAM..$MIRROR")
    behind=$(git rev-list --count "$MIRROR..$UPSTREAM")
    if [ "$ahead" != "0" ]; then
        # The only state that breaks a sync. Behind is normal; ahead is the bug.
        printf '%-34s %s ahead / %s behind  <- CARRIES LOCAL WORK, sync will refuse\n' \
            "$MIRROR" "$ahead" "$behind"
    elif [ "$behind" != "0" ]; then
        printf '%-34s %s behind (sync fast-forwards it)\n' "$MIRROR" "$behind"
    else
        printf '%-34s exact mirror\n' "$MIRROR"
    fi
    echo
    local b
    for b in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
        [ "$b" != "$MIRROR" ] || continue
        ahead=$(git rev-list --count "$UPSTREAM..$b")
        if [ "$(git merge-base "$b" "$UPSTREAM")" = "$(git rev-parse "$UPSTREAM")" ]; then
            printf '%-34s %s commit(s) on current %s\n' "$b" "$ahead" "$UPSTREAM"
        else
            printf '%-34s %s commit(s), on an OLDER base -- needs sync\n' "$b" "$ahead"
        fi
    done
}

cmd_sync() {
    local branch pre_ref old_base
    branch=$(resolve_branch "${1:-}" sync)
    pre_ref="refs/fork-sync/$branch/pre-rebase"

    require_upstream_remote
    [ -z "$(git status --porcelain)" ] \
        || die "Working tree is dirty. Commit first (this script will not stash)."

    assert_mirror_clean

    # The base the branch currently sits on, so `verify` can reconstruct which
    # files we had diverged in BEFORE the rebase moved everything.
    old_base=$(git merge-base "$branch" "$UPSTREAM")
    git update-ref "$pre_ref" "$branch"
    git update-ref "$pre_ref-base" "$old_base"

    echo "==> Fetching $UPSTREAM_REMOTE"
    git fetch "$UPSTREAM_REMOTE"

    assert_mirror_clean

    echo "==> Fast-forwarding $MIRROR to $UPSTREAM"
    git checkout "$MIRROR"
    git merge --ff-only "$UPSTREAM"

    echo "==> Rebasing $branch onto $UPSTREAM"
    git checkout "$branch"
    if ! git rebase "$UPSTREAM"; then
        cat <<EOF

Rebase stopped on a conflict. Resolve it, then:

    git add <files>            # explicit paths; \`git add -A\` is refused here
    git rebase --continue      # or: git rebase --abort

When the rebase finishes, run:

    $0 verify $branch

Resolve toward the FORK'S INTENT: read the commit message of the patch being
applied (\`git log -1 HEAD\` mid-rebase shows the one that landed; \`git rebase
--show-current-patch\` shows the one conflicting) so you know which behaviour
the hunk was protecting, and keep that while taking upstream's refactor around
it. Never resolve by deleting the fork's change to make the build pass -- that
is the silent loss \`verify\` exists to catch.
EOF
        exit 1
    fi

    cmd_verify "$branch"
}

cmd_verify() {
    local branch pre_ref pre_tip pre_base status=0
    branch=$(resolve_branch "${1:-}" verify)
    pre_ref="refs/fork-sync/$branch/pre-rebase"

    git rev-parse --verify --quiet "$pre_ref" >/dev/null \
        || die "No recorded pre-rebase state for $branch. Run 'sync' first."

    pre_tip=$(git rev-parse "$pre_ref")
    pre_base=$(git rev-parse "$pre_ref-base")

    echo "==> Files $branch had diverged in before the rebase, and their fate now"
    local lost=() f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        if git diff --quiet "$UPSTREAM" "$branch" -- "$f"; then
            # No longer differs from upstream. Either upstream absorbed our
            # change, or the rebase dropped it. Only a human can tell which.
            printf '    ABSORBED-OR-LOST  %s\n' "$f"
            lost+=("$f")
        else
            printf '    kept              %s\n' "$f"
        fi
    done < <(divergent_files "$pre_base" "$pre_tip")

    echo
    echo "==> $MIRROR is still a pristine mirror of $UPSTREAM"
    if [ "$(git rev-parse "$MIRROR")" = "$(git rev-parse "$UPSTREAM")" ]; then
        echo "    ok"
    else
        echo "    NO -- $MIRROR has drifted from $UPSTREAM; see '$0 status'" >&2
        status=1
    fi

    echo
    echo "==> $branch sits directly on $UPSTREAM"
    if [ "$(git merge-base "$branch" "$UPSTREAM")" = "$(git rev-parse "$UPSTREAM")" ]; then
        echo "    ok"
    else
        echo "    NO -- $branch is not rebased onto current $UPSTREAM" >&2
        status=1
    fi

    if [ ${#lost[@]} -gt 0 ]; then
        cat >&2 <<EOF

${#lost[@]} file(s) no longer differ from upstream. Check EACH one: a file is
fine if upstream absorbed the same change, and a silent loss if not.

    git show $pre_tip -- <file>        # what we used to have
    git log $UPSTREAM -- <file>        # whether upstream did it too

A rebase reports a conflict only when two sides edit the same region. Where
upstream rewrote a file we had also changed, ours can vanish with no conflict
raised at all, which is why this check reads the file list rather than trusting
the rebase's exit code.
EOF
        status=1
    fi

    return "$status"
}

case "${1:-}" in
    sync)   shift; cmd_sync "${1:-}" ;;
    verify) shift; cmd_verify "${1:-}" ;;
    status) cmd_status ;;
    *)      die "usage: $0 {sync|verify} [branch] | $0 status" ;;
esac
