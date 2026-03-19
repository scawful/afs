;;; afs-emacs.el --- Emacs helpers for AFS briefing and capture -*- lexical-binding: t; -*-

;; Minimal Emacs integration for AFS session startup and morning briefing flows.

;;; Code:

(require 'subr-x)

(defgroup afs-emacs nil
  "Emacs helpers for the Agentic File System."
  :group 'tools)

(defcustom afs-emacs-cli-script
  (expand-file-name "~/src/lab/afs/scripts/afs")
  "Path to the AFS CLI wrapper."
  :type 'file)

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
  (let* ((script (expand-file-name afs-emacs-cli-script))
         (args (append
                (list "briefing" "--org")
                (when skip-gws (list "--no-gws"))))
         (buf (get-buffer-create afs-emacs-briefing-buffer-name))
         (content
          (condition-case err
              (apply #'afs-emacs--run-command-capture script args)
            (error
             (format
              "#+TITLE: AFS Morning Briefing Error\n\n- Command: %s %s\n- Error: %s\n"
              script
              (string-join args " ")
              (error-message-string err))))))
    (with-current-buffer buf
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert content)
        (goto-char (point-min))
        (if (fboundp 'org-mode)
            (org-mode)
          (text-mode))
        (setq buffer-read-only t)
        (local-set-key (kbd "c") #'afs-emacs-briefing-capture)
        (local-set-key (kbd "g") #'afs-emacs-briefing-open)))
    (pop-to-buffer buf)))

(provide 'afs-emacs)
;;; afs-emacs.el ends here
