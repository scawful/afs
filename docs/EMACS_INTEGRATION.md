# Emacs Integration

AFS ships a CLI-backed Emacs helper in `contrib/emacs/afs-emacs.el`. It keeps
the Elisp surface thin while exposing the current AFS session and background
agent workflows directly inside Emacs.

## Included Commands

- `afs-emacs-dispatch`: command palette for the common AFS actions
- `afs-emacs-briefing-open`: render `afs briefing --org`
- `afs-emacs-briefing-capture`: capture a follow-up Org task from the briefing
- `afs-emacs-doctor-open`: render `afs doctor` in a refreshable buffer
- `afs-emacs-agents-status-open`: show `afs agents ps --all`
- `afs-emacs-agents-monitor`: stream `afs agents monitor --all --json`
- `afs-emacs-session-prepare-client-open`: inspect `afs session prepare-client`
- `afs-emacs-session-replay-open`: inspect `afs session replay --json`
- `afs-emacs-chat`: open `hafs chat` from your AFS context

Static AFS buffers share a small UX contract:

- `g`: refresh the buffer by re-running the underlying AFS command
- `q`: quit the window
- `c`: capture a follow-up item when you are in the morning briefing buffer

## Load Path

```elisp
;; Adjust to wherever you cloned AFS
(add-to-list 'load-path "/path/to/afs/contrib/emacs")
(require 'afs-emacs)
```

## Minimal Setup

```elisp
(setq afs-emacs-cli-script "/path/to/afs/scripts/afs")
(setq afs-emacs-capture-file "~/notes/tasks.md")
(setq afs-emacs-default-client "codex")

(global-set-key (kbd "C-c a a") #'afs-emacs-dispatch)
(global-set-key (kbd "C-c a m") #'afs-emacs-briefing-open)
(global-set-key (kbd "C-c a d") #'afs-emacs-doctor-open)
(global-set-key (kbd "C-c a g") #'afs-emacs-agents-status-open)
```

## Spacemacs Example

```elisp
(spacemacs/set-leader-keys
  "oxA" #'afs-emacs-dispatch
  "oxc" #'afs-emacs-chat
  "oxC" #'afs-emacs-briefing-open
  "oxD" #'afs-emacs-doctor-open
  "oxG" #'afs-emacs-agents-status-open
  "oxI" #'afs-emacs-session-prepare-client-open
  "oxM" #'afs-emacs-agents-monitor
  "oxR" #'afs-emacs-session-replay-open)
```

## Notes

- The helper delegates to the normal AFS CLI, so improvements to `afs doctor`,
  `afs session ...`, or `afs agents ...` show up in Emacs automatically.
- `afs-emacs-briefing-capture` requires `afs-emacs-capture-file` to be set.
- `afs-emacs-agents-monitor` uses a live compilation buffer because the command
  streams until interrupted.
