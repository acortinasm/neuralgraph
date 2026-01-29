# Update Documentation After Sprint Completion

When a sprint is completed and merged, update the following documentation files to reflect the new functionality:

## Files to Update

1. **docs/neural_graph_planning.md** - Mark the sprint as completed (change 📅 to ✅)

2. **docs/COMPLETE_FEATURES.md** - Add new features/capabilities:
   - Update version number and date at the top
   - Add new sections for any new functionality
   - Update the Sprint History table at the bottom
   - Update line counts if significant code was added

3. **docs/USER_GUIDE.md** - Add user-facing documentation:
   - Add examples for new query syntax
   - Document new CLI commands
   - Add new sections for major features
   - Update the Quick Reference section

4. **docs/API_REFERENCE.md** - If APIs changed:
   - Document new endpoints
   - Update request/response formats
   - Add new gRPC services

5. **docs/QUICK_REFERENCE.md** - Update cheat sheet:
   - Add new commands
   - Add new NGQL syntax

## Update Process

1. Read the sprint PR/commits to understand what was implemented
2. Update docs/neural_graph_planning.md - mark sprint as ✅
3. Update docs/COMPLETE_FEATURES.md with technical details
4. Update docs/USER_GUIDE.md with user-facing examples
5. Commit changes with message: "docs: update documentation for Sprint XX"
6. Push to main

## Sprint Information to Include

For each completed sprint, document:
- **What**: The feature/capability added
- **Why**: The problem it solves
- **How**: Technical approach (briefly)
- **Usage**: NGQL examples, CLI commands, API calls

## Example Sprint Documentation

For Sprint 53 (Cluster Management):
- Added ClientRequest RPC for leader routing
- Cluster-aware client automatically finds leader
- New metrics: raft_term, raft_log_index, cluster_node_count
- CLI: `neuralgraph cluster info`, `neuralgraph cluster health`

## Notes

- Keep documentation concise and practical
- Include working code examples
- Update version numbers consistently
- Never add Co-Authored-By lines to commits
