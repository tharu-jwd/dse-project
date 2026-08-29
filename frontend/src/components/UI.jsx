import { useEffect, useRef } from 'react'
import Icon from './Icon'

export function Loading({ label = 'Loading…' }) {
  return (
    <div className="loading" role="status">
      <span className="spinner" aria-hidden="true" /> <span>{label}</span>
    </div>
  )
}

export function EmptyState({ icon = 'file', title, message, action }) {
  return (
    <div className="empty-state">
      <div className="empty-state__icon">
        <Icon name={icon} size={26} />
      </div>
      <h3>{title}</h3>
      <p>{message}</p>
      {action}
    </div>
  )
}

export function Alert({ type = 'error', title, children }) {
  return (
    <div className={`alert alert--${type}`} role={type === 'error' ? 'alert' : 'status'}>
      <Icon name={type === 'success' ? 'check' : 'alert'} />
      <div>
        {title && <strong>{title}</strong>}
        {children && <div>{children}</div>}
      </div>
    </div>
  )
}

export function StatusBadge({ status }) {
  const labels = {
    DRAFT: 'Draft',
    FINALIZED: 'Finalized',
    PROCESSING: 'Processing',
    FAILED: 'Failed',
    PUBLISHED: 'Published',
    SUBMITTED: 'Submitted',
    REVIEWED: 'Reviewed',
    NOT_STARTED: 'Not started',
    COMPLETED: 'Completed',
    UPLOADING: 'Uploading',
  }
  return (
    <span className={`badge badge--${status.toLowerCase().replace('_', '-')}`}>
      {labels[status] || status}
    </span>
  )
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = 'Confirm',
  dangerous = false,
  busy = false,
  onConfirm,
  onCancel,
}) {
  const cancelRef = useRef(null)
  useEffect(() => {
    if (open) cancelRef.current?.focus()
  }, [open])
  if (!open) return null
  return (
    <div
      className="dialog-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && onCancel()}
    >
      <div
        className="dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="dialog-title"
        aria-describedby="dialog-message"
      >
        <div className="dialog__icon">
          <Icon name={dangerous ? 'alert' : 'check'} />
        </div>
        <h2 id="dialog-title">{title}</h2>
        <p id="dialog-message">{message}</p>
        <div className="dialog__actions">
          <button
            ref={cancelRef}
            className="button button--secondary"
            onClick={onCancel}
            disabled={busy}
          >
            Cancel
          </button>
          <button
            className={`button ${dangerous ? 'button--danger' : 'button--primary'}`}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? 'Working…' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

export function PageHeader({ eyebrow, title, description, actions, back }) {
  return (
    <header className="page-header">
      {back}
      {eyebrow && <span className="eyebrow">{eyebrow}</span>}
      <div className="page-header__row">
        <div>
          <h1>{title}</h1>
          {description && <p>{description}</p>}
        </div>
        {actions && <div className="page-header__actions">{actions}</div>}
      </div>
    </header>
  )
}

export function ProgressSteps({ steps, current }) {
  return (
    <ol className="progress-steps" aria-label="Progress">
      {steps.map((step, index) => (
        <li
          key={step}
          className={index < current ? 'complete' : index === current ? 'current' : ''}
        >
          <span>{index < current ? '✓' : index + 1}</span>
          <em>{step}</em>
        </li>
      ))}
    </ol>
  )
}
