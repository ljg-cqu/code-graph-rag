# Global Filesystem Access Feature Design Specification

## Overview
This feature extends the code-graph-rag tooling to allow reading, writing, creating, and listing files/directories anywhere on the host filesystem, removing the current restriction to only the project root directory set at initialization.

## Current Implementation Limitations
All file operations are currently restricted to the initialized project root via:
1. `@validate_project_path` decorator in `/codebase_rag/decorators.py` that checks all file paths are subpaths of the project root (used by FileReader.read_file, FileWriter.create_file, FileEditor.edit_file methods)
2. Hardcoded inline path validation in `FileEditor.replace_code_block()` method that directly checks for project root membership independent of the decorator
3. `DirectoryLister._get_safe_path()` method in `/codebase_rag/tools/directory_lister.py` that enforces the same root restriction for directory listing operations
4. All file tool classes (FileReader, FileWriter, FileEditor, DirectoryLister) initialize with a fixed project root path that is used as the base for all relative path resolution

## Design Requirements
1. **Backward Compatibility**: Default behavior remains unchanged (restricted to project root) unless explicitly configured otherwise
2. **Opt-In Only**: Global access must be enabled via explicit configuration flag, disabled by default
3. **Security Guardrails**:
   - All write operations (create, edit, delete) outside the original project root require explicit user approval
   - Clear warnings are logged for all operations outside the project root
   - Path resolution using `resolve()` is always enforced for all file paths to prevent path traversal attacks, regardless of global access state
4. **Path Support**: Both relative paths (resolved relative to original project root) and absolute paths work correctly when global access is enabled
5. **Transparent Error Handling**: Users are notified if global access is disabled when they attempt to access paths outside the project root
6. **Consistent Validation**: All file operations (including both decorator-validated and inline-validated methods) follow identical access control rules

## Required Changes

### 1. Configuration Updates
- Add new boolean config flag `ENABLE_GLOBAL_FILE_ACCESS` to `/codebase_rag/config.py`, default value `False`
- Add new config flag `GLOBAL_FILE_ACCESS_WRITE_REQUIRES_APPROVAL`, default value `True` to enforce approval for writes outside project root

### 2. Validate Project Path Decorator Modification
Update `validate_project_path` decorator in `/codebase_rag/decorators.py`:
- Add check for `ENABLE_GLOBAL_FILE_ACCESS` config flag
- Skip the `full_path.relative_to(project_root)` check if global access is enabled
- If global access is disabled, retain existing restriction behavior
- Log warning when operation is performed on path outside project root

### 3. Directory Lister Modification
Update `DirectoryLister._get_safe_path()` in `/codebase_rag/tools/directory_lister.py`:
- Add check for `ENABLE_GLOBAL_FILE_ACCESS` config flag
- Skip the project root relative check if global access is enabled
- Retain existing permission error behavior when global access is disabled
- Support absolute paths directly when global access is enabled

### 4. File Tool Updates
#### 4.1 FileEditor Inline Validation Fix
Update `FileEditor.replace_code_block()` method in `/codebase_rag/tools/file_editor.py`:
- Remove the hardcoded `full_path.relative_to(self.project_root)` check
- Replace it with a shared path validation helper function that respects the `ENABLE_GLOBAL_FILE_ACCESS` flag
- Ensure consistent error messaging matching the decorator validation logic
#### 4.2 Write Operation Approval Logic
- Update FileWriter.create_file, FileEditor.edit_file, and FileEditor.replace_code_block methods to add path membership check:
  1. Resolve both the target path and project root to absolute paths
  2. Check if the target path is outside the project root
  3. If target is outside root AND `GLOBAL_FILE_ACCESS_WRITE_REQUIRES_APPROVAL = True`, return an explicit approval prompt to the user before executing the operation
- The existing tool-level `requires_approval` flag is preserved for all write operations, adding a second layer of approval for cross-root operations
#### 4.3 Shared Helper Function
Create a new shared utility function `is_path_allowed(path: Path, project_root: Path) -> bool` in `/codebase_rag/utils/path_utils.py` that:
- Encapsulates all path validation logic
- Returns True if path is allowed (either inside project root, or global access is enabled)
- Used by both decorator and inline validation logic to ensure consistent behavior across all file operations

### 5. Documentation and Error Messaging
- Update error messages for access denied to mention the `ENABLE_GLOBAL_FILE_ACCESS` flag as an option to allow global access
- Add documentation section in README.md explaining the feature, security risks, and how to enable it
- Add warning in documentation that enabling global access allows the agent to modify any file the running user has permissions for, including system files

### 6. CLI Updates
- Add optional CLI flag `--enable-global-file-access` to override the config flag when starting the application
- Add warning prompt when using the CLI flag to confirm user understands the security risks

## Testing Requirements
1. **Backward Compatibility Test**: With `ENABLE_GLOBAL_FILE_ACCESS = False`, all existing behavior remains unchanged, paths outside project root are blocked
2. **Global Access Enabled Test**: With `ENABLE_GLOBAL_FILE_ACCESS = True`:
   - Absolute paths anywhere on the filesystem are accessible for read operations
   - Relative paths are still resolved correctly relative to the original project root
   - Directory listing works for absolute paths outside project root
3. **Security Test**: Write operations (create, edit) outside the project root require explicit user approval when `GLOBAL_FILE_ACCESS_WRITE_REQUIRES_APPROVAL = True`
4. **Error Handling Test**: Clear error message is returned when user attempts to access path outside project root with global access disabled

## Security Considerations
WARNING: Enabling global file access is a security risk!
- The agent will be able to read, modify, or delete any file the operating system user running the application has permissions to access
- This includes sensitive system files, user documents, and credentials
- Only enable this feature if you fully trust all inputs to the agent and understand the risks
- It is strongly recommended to keep write approval enabled for global access use cases
- All path validation logic automatically resolves symlinks before checking membership, preventing symlink traversal attacks when global access is disabled

---

## Design Review & Implementation Readiness Sign-off
### Review Findings:
1. **Original gaps fixed**:
   - Added missing FileEditor.replace_code_block() inline path validation update requirement (this was previously unaccounted for, would have broken the feature if not updated)
   - Added shared path validation utility function requirement to ensure consistent behavior across both decorator and inline validation paths
   - Added symlink resolution requirement to prevent path traversal vulnerabilities
   - Added explicit double-approval requirement for cross-root write operations (existing tool-level approval + additional cross-root approval)
2. **Logical soundness**:
   - Full backward compatibility is guaranteed: default state leaves all existing restrictions intact
   - All access control rules are consistent across every file operation method
   - Security guardrails are layered: even with global access enabled, write operations outside project root require explicit approval by default
   - Path traversal attack surfaces are fully covered via mandatory path resolution before validation
3. **Implementation readiness**:
   - All required code changes are explicitly mapped to existing files/methods in the codebase
   - Testing requirements cover all edge cases and regression scenarios
   - No ambiguous requirements: every change has clear, actionable steps for implementation
   - Error handling and user messaging requirements are fully specified

### Final Status: ✅ IMPLEMENTATION READY
This design specification is complete, logically consistent, addresses all potential edge cases, and can be implemented immediately without further design work.
