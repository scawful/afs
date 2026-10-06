;;; afs-emacs.el --- Emacs helpers for AFS sessions, briefings, and agents -*- lexical-binding: t; -*-

;; Thin Emacs integration for common AFS session, replay, and background-agent
;; workflows. The helper stays CLI-backed so new AFS surfaces show up in Emacs
;; without duplicating logic in Elisp.

;;; Code:

(require 'json)
(require 'subr-x)

(defgroup afs-emacs nil
  "Emacs helpers for the Agentic File System."
  :group 'tools)

(defcustom afs-emacs-cli-script
  (or (executable-find "afs") "afs")
  "AFS executable name or path.  Defaults to `afs' on PATH."
  :type 'string)

(defcustom afs-emacs-briefing-buffer-name
  "*AFS Morning Briefing*"
  "Buffer name used for rendered AFS briefings."
  :type 'string)

(defcustom afs-emacs-capture-key
  "A"
  "Org capture key used for AFS follow-up items."
  :type 'string)

(defcustom afs-emacs-capture-file
  nil
  "Optional Org file used for AFS follow-up capture.
When nil, no capture template is installed."
  :type '(choice (const :tag "Disabled" nil) file))

(defcustom afs-emacs-capture-headline
  "Inbox"
  "Headline used when capturing AFS follow-up items."
  :type 'string)

(defcustom afs-emacs-default-client
  "codex"
  "Default client label for `afs session prepare-client`."
  :type 'string)

(defcustom afs-emacs-chat-command
  "hafs chat"
  "Shell command launched by `afs-emacs-chat`."
  :type 'string)

(defcustom afs-emacs-chat-directory
  (expand-file-name "~/.context")
  "Working directory used for `afs-emacs-chat`."
  :type 'directory)

(defvar-local afs-emacs--rerender-fn nil
  "Buffer-local refresh function for rendered AFS output.")

(defvar afs-emacs-command-mode-map
  (let ((map (make-sparse-keymap)))
    (set-keymap-parent map special-mode-map)
    (define-key map (kbd "g") #'afs-emacs-refresh-buffer)
    (define-key map (kbd "q") #'quit-window)
    map)
  "Keymap for `afs-emacs-command-mode`.")

(define-derived-mode afs-emacs-command-mode special-mode "AFS"
  "Major mode for static AFS command output buffers."
  (setq truncate-lines nil))

(defconst afs-emacs--known-clients
  '("codex" "claude" "gemini" "generic")
  "Known AFS model/client profiles.")

(defconst afs-emacs--dispatch-commands
  '(("Morning briefing" . afs-emacs-briefing-open)
    ("AFS doctor" . afs-emacs-doctor-open)
    ("Background agents" . afs-emacs-agents-status-open)
    ("Monitor background agents" . afs-emacs-agents-monitor)
    ("Prepare client payload" . afs-emacs-session-prepare-client-open)
    ("Replay session" . afs-emacs-session-replay-open)
    ("Model chat" . afs-emacs-chat))
  "Command palette entries for `afs-emacs-dispatch`.")

(defun afs-emacs--run-command-capture (program &rest args)
  "Run PROGRAM with ARGS and return stdout as a string."
  (with-temp-buffer
    (let ((exit-code (apply #'process-file program nil t nil args)))
      (if (zerop exit-code)
          (string-trim-right (buffer-string))
        (error "Command failed (%s %s): %s"
               program
               (string-join args " ")
               (buffer-string))))))

(defun afs-emacs--json-parse-safe (json-string)
  "Parse JSON-STRING and return Lisp data, or nil when parsing fails."
  (condition-case nil
      (if (fboundp 'json-parse-string)
          (json-parse-string json-string
                             :object-type 'alist
                             :array-type 'list
                             :null-object nil
                             :false-object nil)
        (let ((json-object-type 'alist)
              (json-array-type 'list)
              (json-null nil)
              (json-false nil))
          (json-read-from-string json-string)))
    (error nil)))

(defun afs-emacs--alist-value (alist key)
  "Return ALIST value for KEY, supporting symbol or string keys."
  (or (alist-get key alist)
      (and (symbolp key) (alist-get (symbol-name key) alist nil nil #'equal))
      (and (stringp key) (alist-get (intern key) alist))))

(defun afs-emacs--session-id-candidates ()
  "Return recent AFS session IDs for completion."
  (condition-case nil
      (let* ((raw (afs-emacs--run-command-capture
                   (expand-file-name afs-emacs-cli-script)
                   "session" "replay" "list" "--limit" "20" "--json"))
             (items (afs-emacs--json-parse-safe raw)))
        (delq nil
              (mapcar (lambda (entry)
                        (afs-emacs--alist-value entry 'session_id))
                      items)))
    (error nil)))

(defun afs-emacs--read-session-id (&optional prompt)
  "Prompt for a recent session id using PROMPT."
  (let* ((candidates (afs-emacs--session-id-candidates))
         (default (or (getenv "AFS_SESSION_ID") (car candidates)))
         (prompt (or prompt "Session id: ")))
    (if candidates
        (completing-read
         (if default
             (format "%s(default %s): " prompt default)
           prompt)
         candidates
         nil
         nil
         nil
         nil
         default)
      (read-string prompt nil nil default))))

(defun afs-emacs--show-command-error (title program args err)
  "Return an error document for TITLE when PROGRAM ARGS raises ERR."
  (format
   "# %s Error\n\n- Command: %s %s\n- Error: %s\n"
   title
   program
   (string-join args " ")
   (error-message-string err)))

(defun afs-emacs-refresh-buffer ()
  "Refresh the current rendered AFS buffer."
  (interactive)
  (unless (functionp afs-emacs--rerender-fn)
    (user-error "This buffer is not backed by a refreshable AFS command"))
  (funcall afs-emacs--rerender-fn))

(defun afs-emacs--prepare-buffer (buffer mode refresh-fn setup-fn)
  "Prepare BUFFER using MODE, REFRESH-FN, and optional SETUP-FN."
  (pcase mode
    ('org
     (if (fboundp 'org-mode)
         (org-mode)
       (text-mode)))
    ('json
     (text-mode)
     (when (fboundp 'json-pretty-print-buffer)
       (condition-case nil
           (json-pretty-print-buffer)
         (error nil))))
    (_
     (afs-emacs-command-mode)))
  (setq-local afs-emacs--rerender-fn refresh-fn)
  (local-set-key (kbd "g") #'afs-emacs-refresh-buffer)
  (local-set-key (kbd "q") #'quit-window)
  (when (functionp setup-fn)
    (funcall setup-fn))
  (setq buffer-read-only t)
  buffer)

(defun afs-emacs--show-command-output (buffer-name args &rest options)
  "Run AFS CLI ARGS and show BUFFER-NAME using OPTIONS.
Supported OPTIONS keys are:

- `:mode` one of `org`, `json`, or `text`
- `:title` human-readable title for error output
- `:setup` buffer-local setup function"
  (let* ((script (expand-file-name afs-emacs-cli-script))
         (buffer (get-buffer-create buffer-name))
         (mode (or (plist-get options :mode) 'text))
         (title (or (plist-get options :title) "AFS Command"))
         (setup-fn (plist-get options :setup))
         (refresh-fn (lambda ()
                       (apply #'afs-emacs--show-command-output
                              (append (list buffer-name args) options))))
         (content
          (condition-case err
              (apply #'afs-emacs--run-command-capture script args)
            (error
             (afs-emacs--show-command-error title script args err)))))
    (with-current-buffer buffer
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert content)
        (goto-char (point-min))
        (afs-emacs--prepare-buffer buffer mode refresh-fn setup-fn)))
    (pop-to-buffer buffer)))

(defun afs-emacs--shell-command (buffer-name args &optional directory)
  "Run AFS CLI ARGS in a live compilation buffer named BUFFER-NAME.
When DIRECTORY is non-nil, use it as `default-directory`."
  (let* ((script (expand-file-name afs-emacs-cli-script))
         (default-directory (expand-file-name (or directory default-directory)))
         (command (mapconcat #'shell-quote-argument (cons script args) " ")))
    (compilation-start command
                       'compilation-mode
                       (lambda (_) buffer-name))))

(defun afs-emacs--start-shell-command (buffer-name command &optional directory)
  "Run shell COMMAND in BUFFER-NAME using DIRECTORY as `default-directory`."
  (let ((default-directory (expand-file-name (or directory default-directory))))
    (compilation-start command
                       'compilation-mode
                       (lambda (_) buffer-name))))

(defun afs-emacs-register-capture-template ()
  "Append the AFS follow-up capture template when configured."
  (interactive)
  (when (and (boundp 'org-capture-templates)
             (stringp afs-emacs-capture-file)
             (not (assoc afs-emacs-capture-key org-capture-templates)))
    (add-to-list
     'org-capture-templates
     `(
       ,afs-emacs-capture-key
       "AFS Follow-up"
       entry
       (file+headline ,afs-emacs-capture-file ,afs-emacs-capture-headline)
       "* TODO [AFS] %?\n:PROPERTIES:\n:Created: %U\n:Source: AFS Morning Briefing\n:END:\n%i\n%a\n")
     t)))

(with-eval-after-load 'org-capture
  (afs-emacs-register-capture-template))

(defun afs-emacs-briefing-capture ()
  "Capture a follow-up item from the AFS morning briefing."
  (interactive)
  (unless afs-emacs-capture-file
    (user-error "Set `afs-emacs-capture-file` before capturing AFS follow-up items"))
  (require 'org-capture)
  (afs-emacs-register-capture-template)
  (org-capture nil afs-emacs-capture-key))

(defun afs-emacs-briefing-open (&optional skip-gws)
  "Render `afs briefing --org` in an Org buffer.
With prefix argument SKIP-GWS, add `--no-gws` to the command."
  (interactive "P")
  (afs-emacs--show-command-output
   afs-emacs-briefing-buffer-name
   (append
    (list "briefing" "--org")
    (when skip-gws (list "--no-gws")))
   :mode 'org
   :title "AFS Morning Briefing"
   :setup (lambda ()
            (local-set-key (kbd "c") #'afs-emacs-briefing-capture))))

(defun afs-emacs-chat ()
  "Start interactive `hafs chat` from `afs-emacs-chat-directory`."
  (interactive)
  (let ((default-directory (expand-file-name afs-emacs-chat-directory)))
    (cond
     ((and (fboundp 'vterm) (fboundp 'vterm-send-string) (fboundp 'vterm-send-return))
      (vterm t)
      (vterm-send-string afs-emacs-chat-command)
      (vterm-send-return))
     ((fboundp 'eshell)
     (eshell t)
      (goto-char (point-max))
      (insert afs-emacs-chat-command)
      (eshell-send-input))
     (t
      (afs-emacs--start-shell-command "*AFS Chat*"
                                      afs-emacs-chat-command
                                      afs-emacs-chat-directory)))))

(defun afs-emacs-doctor-open (&optional fix)
  "Render `afs doctor` in a refreshable buffer.
With prefix argument FIX, add `--fix`."
  (interactive "P")
  (afs-emacs--show-command-output
   "*AFS Doctor*"
   (append
    (list "doctor")
    (when fix (list "--fix")))
   :title "AFS Doctor"))

(defun afs-emacs-agents-status-open ()
  "Render `afs agents ps --all` in a refreshable buffer."
  (interactive)
  (afs-emacs--show-command-output
   "*AFS Agents*"
   (list "agents" "ps" "--all")
   :title "AFS Agents"))

(defun afs-emacs-agents-monitor (&optional session-id)
  "Start `afs agents monitor --all --json`.
With prefix argument, prompt for SESSION-ID and monitor that session."
  (interactive
   (list
    (when current-prefix-arg
      (afs-emacs--read-session-id "Monitor session id: "))))
  (afs-emacs--shell-command
   "*AFS Agent Monitor*"
   (append
    (list "agents" "monitor" "--all" "--json")
    (unless (string-empty-p (or session-id ""))
      (list "--session-id" session-id)))))

(defun afs-emacs-session-prepare-client-open (client query task)
  "Render `afs session prepare-client` for CLIENT with QUERY and TASK."
  (interactive
   (list
    (completing-read "Client: " afs-emacs--known-clients nil t nil nil afs-emacs-default-client)
    (read-string "Query (optional): ")
    (read-string "Task (optional): ")))
  (let ((model (if (member client afs-emacs--known-clients) client "generic")))
    (afs-emacs--show-command-output
     (format "*AFS Prepare: %s*" client)
     (append
      (list "session" "prepare-client"
            "--client" client
            "--model" model
            "--path" (expand-file-name default-directory)
            "--json")
      (unless (string-empty-p query)
        (list "--query" query))
      (unless (string-empty-p task)
        (list "--task" task)))
     :mode 'json
     :title (format "AFS Prepare Client (%s)" client))))

(defun afs-emacs-session-replay-open (&optional session-id)
  "Render `afs session replay --json` for SESSION-ID."
  (interactive
   (list
    (afs-emacs--read-session-id "Replay session id: ")))
  (let ((session-id (or session-id "")))
    (if (string-empty-p session-id)
        (user-error "Session id is required for replay")
      (afs-emacs--show-command-output
       (format "*AFS Replay: %s*" session-id)
       (list "session" "replay" "--session-id" session-id "--json")
       :mode 'json
       :title (format "AFS Session Replay (%s)" session-id)))))

(defun afs-emacs-dispatch ()
  "Open a command palette for the most common AFS Emacs actions."
  (interactive)
  (let* ((choice (completing-read
                  "AFS command: "
                  (mapcar #'car afs-emacs--dispatch-commands)
                  nil
                  t))
         (command (cdr (assoc choice afs-emacs--dispatch-commands))))
    (call-interactively command)))

(provide 'afs-emacs)
;;; afs-emacs.el ends here
