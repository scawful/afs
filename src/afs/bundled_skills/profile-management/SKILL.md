---
name: profile-management
triggers:
  - profile
  - switch
  - bundle
  - extension
profiles:
  - general
requires:
  - afs
---

# Profile Management

Profiles control which knowledge, skills, tools, and agents are active.

## Commands

| Command | Description |
|---------|-------------|
| `afs profile current` | Show active profile with resolved mounts |
| `afs profile switch <name>` | Switch to a different profile |
| `afs profile list` | List available profiles |
| `afs bundle pack <profile>` | Package a profile as a portable bundle |
| `afs bundle install <path>` | Install a bundle as an extension |
| `afs bundle inspect <path>` | Inspect bundle contents |

## Profile Inheritance

Profiles can extend other profiles:

```toml
[profiles.dev]
inherits = ["base"]
knowledge_mounts = ["~/knowledge/dev"]
skill_roots = ["~/skills/dev"]
```

Child profiles inherit and override parent settings.

## Extensions

Extensions add knowledge, skills, CLI commands, MCP tools, and policies:

```
extensions/my-ext/
  extension.toml    # manifest
  knowledge/        # knowledge files
  skills/           # skill directories
  hooks/            # policy enforcement
```

Enable in config: `enabled_extensions = ["my-ext"]`

## Tips

- Use `afs profile current --json` for full resolved state
- Profiles are defined in `~/.config/afs/config.toml` under `[profiles.<name>]`
- The `general` profile is implicit when no profile is set
- Bundled repo skills are auto-discovered from `AFS_ROOT/skills` when present
