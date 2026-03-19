# Emacs Integration

AFS ships a small Emacs helper for the morning briefing flow:

- file: `contrib/emacs/afs-emacs.el`
- primary command: `afs-emacs-briefing-open`
- capture helper: `afs-emacs-briefing-capture`

The helper renders `afs briefing --org` into an Org buffer and can install an
Org capture template for follow-up items.

## Load Path

```elisp
(add-to-list 'load-path "~/src/lab/afs/contrib/emacs")
(require 'afs-emacs)
```

## Minimal Setup

```elisp
(setq afs-emacs-cli-script "~/src/lab/afs/scripts/afs")
(setq afs-emacs-capture-file "~/Journal/inbox.org")

(global-set-key (kbd "C-c a m") #'afs-emacs-briefing-open)
```

With that in place:

- `M-x afs-emacs-briefing-open` renders `afs briefing --org`
- `C-u M-x afs-emacs-briefing-open` renders `afs briefing --org --no-gws`
- `c` inside the briefing buffer starts an `AFS Follow-up` Org capture
- `g` inside the briefing buffer refreshes the buffer

## Spacemacs Example

```elisp
(spacemacs/set-leader-keys
  "orm" #'afs-emacs-briefing-open)
```

If you want this to replace an existing “morning report” function, bind that
key to `afs-emacs-briefing-open` instead of calling your older report helper.

## Notes

- The helper is intentionally thin. It does not try to manage your full Emacs
  workflow or journal structure.
- `afs-emacs-briefing-capture` requires `afs-emacs-capture-file` to be set.
- The briefing content comes from the normal AFS CLI, so CLI updates to
  `afs briefing --org` are reflected automatically in Emacs.
