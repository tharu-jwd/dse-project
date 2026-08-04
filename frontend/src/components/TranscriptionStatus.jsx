import Icon from './Icon'

const states = {
  UPLOADING: ['Uploading your media', 'Your file is being securely uploaded.', 25],
  PROCESSING: ['Creating your Sinhala transcript', 'This usually takes a moment in demo mode.', 65],
  COMPLETED: ['Transcript is ready', 'Opening the transcript editor…', 100],
  FAILED: ['Transcription failed', 'We could not process this recording.', 100],
}

export default function TranscriptionStatus({ status = 'UPLOADING', message, onRetry }) {
  const [title, detail, percent] = states[status] || states.PROCESSING
  return (
    <section
      className={`transcription-status transcription-status--${status.toLowerCase()}`}
      aria-live="polite"
      aria-busy={status === 'UPLOADING' || status === 'PROCESSING'}
    >
      <div className="transcription-status__visual">
        {status === 'COMPLETED' ? (
          <Icon name="check" size={28} />
        ) : status === 'FAILED' ? (
          <Icon name="alert" size={28} />
        ) : (
          <span className="wave" aria-hidden="true">
            {[1, 2, 3, 4, 5, 6, 7].map((item) => (
              <i key={item} />
            ))}
          </span>
        )}
      </div>
      <h2>{title}</h2>
      <p>{message || detail}</p>
      <div
        className="progress-track"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <span style={{ width: `${percent}%` }} />
      </div>
      <small>
        {status === 'PROCESSING'
          ? 'Please keep this page open'
          : status === 'UPLOADING'
            ? `${percent}% complete`
            : ''}
      </small>
      {status === 'FAILED' && onRetry && (
        <button type="button" className="button button--primary" onClick={onRetry}>
          Try again
        </button>
      )}
    </section>
  )
}
