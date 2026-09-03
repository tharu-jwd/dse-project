import { useRef, useState } from 'react'
import { useLanguage } from '../contexts/LanguageContext'
import { translate } from '../i18n/translations'
import Icon from './Icon'

export const MAX_FILE_SIZE = 100 * 1024 * 1024
export const formatFileSize = (bytes) =>
  bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`

export function validateMediaFile(
  file,
  { audioOnly = false, maxSize = MAX_FILE_SIZE, language = 'en' } = {},
) {
  if (!file) return translate('upload.chooseFileToContinue', language)
  const validType = audioOnly
    ? file.type.startsWith('audio/')
    : file.type.startsWith('audio/') || file.type.startsWith('video/')
  if (!validType)
    return translate(
      audioOnly ? 'upload.chooseSupportedAudio' : 'upload.chooseSupportedAudioVideo',
      language,
    )
  if (file.size > maxSize)
    return translate('upload.fileTooLarge', language, Math.round(maxSize / 1024 / 1024))
  return ''
}

export default function FileUpload({
  file,
  onChange,
  audioOnly = false,
  error: externalError,
  id = 'media-file',
  icon = 'upload',
  iconImage,
  floatingIcon = false,
  heading,
  tagline,
}) {
  const { t, language } = useLanguage()
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [localError, setLocalError] = useState('')
  const accept = audioOnly ? 'audio/*' : 'audio/*,video/*'
  const choose = (selected) => {
    const error = validateMediaFile(selected, { audioOnly, language })
    setLocalError(error)
    if (!error) onChange(selected)
  }
  const drop = (event) => {
    event.preventDefault()
    setDragging(false)
    choose(event.dataTransfer.files[0])
  }
  if (file)
    return (
      <div className="selected-file">
        <div className="selected-file__icon">
          <Icon name={file.type.startsWith('video/') ? 'play' : 'file'} />
        </div>
        <div>
          <strong>{file.name}</strong>
          <span>
            {file.type || t('upload.mediaFile')} · {formatFileSize(file.size)}
          </span>
        </div>
        <button
          type="button"
          className="button button--text button--danger-text"
          onClick={() => {
            onChange(null)
            setLocalError('')
            if (inputRef.current) inputRef.current.value = ''
          }}
        >
          <Icon name="trash" size={17} /> {t('upload.remove')}
        </button>
      </div>
    )
  return (
    <div>
      <div
        className={`dropzone ${dragging ? 'dropzone--active' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <input
          ref={inputRef}
          id={id}
          type="file"
          accept={accept}
          onChange={(e) => choose(e.target.files[0])}
          aria-describedby={`${id}-hint ${id}-error`}
        />
        <div className={`dropzone__icon ${floatingIcon ? 'dropzone__icon--floating' : ''}`}>
          {iconImage ? (
            <img src={iconImage} alt="" className="dropzone__icon-image" />
          ) : (
            <Icon name={icon} size={icon === 'rocket' ? 40 : 25} />
          )}
        </div>
        {heading && <strong className="dropzone__heading">{heading}</strong>}
        {tagline && <p className="dropzone__tagline">{tagline}</p>}
        <label htmlFor={id}>
          <strong>{t('upload.chooseFile')}</strong> {t('upload.orDragDrop')}
        </label>
        <span id={`${id}-hint`}>
          {audioOnly ? t('upload.audioFormats') : t('upload.audioVideoFormats')} ·{' '}
          {t('upload.upTo100mb')}
        </span>
      </div>
      {(localError || externalError) && (
        <p className="field-error" id={`${id}-error`} role="alert">
          <Icon name="alert" size={16} />
          {localError || externalError}
        </p>
      )}
    </div>
  )
}
