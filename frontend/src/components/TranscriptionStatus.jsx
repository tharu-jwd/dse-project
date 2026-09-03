import { useLanguage } from '../contexts/LanguageContext'
import Icon from './Icon'

const states = {
  UPLOADING: ['transcription.uploadingTitle', 'transcription.uploadingDetail', 25],
  PROCESSING: ['transcription.processingTitle', 'transcription.processingDetail', 65],
  COMPLETED: ['transcription.completedTitle', 'transcription.completedDetail', 100],
  FAILED: ['transcription.failedTitle', 'transcription.failedDetail', 100],
}

export default function TranscriptionStatus({ status = 'UPLOADING', message, onRetry }) {
  const { t } = useLanguage()
  const [titleKey, detailKey, percent] = states[status] || states.PROCESSING
  const title = t(titleKey)
  const detail = t(detailKey)
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
          ? t('transcription.keepPageOpen')
          : status === 'UPLOADING'
            ? t('transcription.percentComplete', percent)
            : ''}
      </small>
      {status === 'FAILED' && onRetry && (
        <button type="button" className="button button--primary" onClick={onRetry}>
          {t('transcription.tryAgain')}
        </button>
      )}
    </section>
  )
}
