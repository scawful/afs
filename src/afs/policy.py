"""AFS permission policy enforcement."""

from __future__ import annotations

from .models import MountType
from .schema import DirectoryConfig, PolicyType


class PolicyEnforcer:
    """Enforces AFS directory policies."""

    def __init__(self, directories: list[DirectoryConfig]):
        self._policies: dict[MountType, PolicyType] = {}
        for directory in directories:
            role_name = directory.role.value if directory.role else directory.name
            try:
                mount_type = MountType(role_name)
            except ValueError:
                continue
            self._policies[mount_type] = directory.policy

    def get_policy(self, mount_type: MountType) -> PolicyType:
        return self._policies.get(mount_type, PolicyType.READ_ONLY)

    def can_read(self, mount_type: MountType) -> bool:
        return True

    def can_write(self, mount_type: MountType) -> bool:
        policy = self.get_policy(mount_type)
        return policy in (PolicyType.WRITABLE, PolicyType.EXECUTABLE)

    def can_execute(self, mount_type: MountType) -> bool:
        return self.get_policy(mount_type) == PolicyType.EXECUTABLE

    def validate_operation(self, mount_type: MountType, operation: str) -> tuple[bool, str]:
        policy = self.get_policy(mount_type)

        if operation == "read":
            return (True, "")

        if operation == "write":
            if policy in (PolicyType.WRITABLE, PolicyType.EXECUTABLE):
                return (True, "")
            return (
                False,
                f"{mount_type.value} is {policy.value}, writing not allowed",
            )

        if operation == "execute":
            if policy == PolicyType.EXECUTABLE:
                return (True, "")
            return (
                False,
                f"{mount_type.value} is {policy.value}, execution not allowed",
            )

        return (False, f"Unknown operation: {operation}")

    def get_policy_description(self, mount_type: MountType) -> str:
        policy = self.get_policy(mount_type)
        descriptions = {
            PolicyType.READ_ONLY: "Read-only (no modifications allowed)",
            PolicyType.WRITABLE: "Writable (modifications allowed)",
            PolicyType.EXECUTABLE: "Executable (can run scripts/binaries)",
        }
        return descriptions.get(policy, "Unknown policy")
