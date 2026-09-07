# From Prompt to Pull Request

A concise checklist for contributors moving a change through the GitHub workflow.

## Clone

1. Fork the repository and clone your fork.
2. Add the upstream remote:

   ```bash
   git remote add upstream https://github.com/HatcherLabs/hatcher-neural-mesh.git
   ```

## Branch

3. Sync latest main:

   ```bash
   git checkout main && git pull upstream main
   ```

4. Create a feature branch:

   ```bash
   git checkout -b <feature-branch>
   ```

## Edit

5. Make focused changes; keep the diff small and scoped to one concern.

## Verify

6. Run formatting check:

   ```bash
   cargo fmt --check
   ```

7. Run the test suite:

   ```bash
   cargo test
   ```

## Commit

8. Stage and commit with a clear conventional message:

   ```bash
   git add <changed-files>
   git commit -m "type: concise summary"
   ```

## Push

9. Push the branch:

   ```bash
   git push -u origin <feature-branch>
   ```

## Pull Request

10. Open a pull request against main. Summarize the change, link related issues, and list manual verification steps before requesting review.
