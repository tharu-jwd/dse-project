// Central translation dictionary. Add a key here, then use it anywhere with t('key.name').
// Keep keys namespaced by area (nav.*, settings.*, help.*, ...) so it stays easy to scan.
export const translations = {
  'nav.dashboard': { en: 'Dashboard', si: 'උපකරණ පුවරුව' },
  'nav.lectureCaptioning': { en: 'Lecture captioning', si: 'දේශන සිරස්තල' },
  'nav.uploadLecture': { en: 'Upload lecture', si: 'දේශනය උඩුගත කරන්න' },
  'nav.myQuizzes': { en: 'My quizzes', si: 'මගේ ප්‍රශ්නාවලි' },
  'nav.manageQuizzes': { en: 'Manage quizzes', si: 'ප්‍රශ්නාවලි කළමනාකරණය' },
  'nav.selfStudyNotes': { en: 'Self-study notes', si: 'ස්වයං අධ්‍යයන සටහන්' },
  'nav.transcriptLibrary': { en: 'Transcript library', si: 'පිටපත් ලේඛනාගාරය' },
  'nav.reviewSubmissions': { en: 'Review submissions', si: 'ඉදිරිපත් කිරීම් සමාලෝචනය' },
  'nav.settings': { en: 'Settings', si: 'සැකසුම්' },
  'nav.quickStartHelp': { en: 'Quick start & help', si: 'ඉක්මන් මඟපෙන්වීම සහ උදව්' },
  'nav.signOut': { en: 'Sign out', si: 'පිටවන්න' },
  'nav.newAdventure': { en: 'New adventure', si: 'නව අත්දැකීමක්' },
  'nav.notifications': { en: 'Notifications', si: 'දැනුම්දීම්' },
  'nav.accountSettings': { en: 'Account settings', si: 'ගිණුම් සැකසුම්' },
  'nav.searchPlaceholder': { en: 'Search your transcripts…', si: 'ඔබේ පිටපත් සොයන්න…' },
  'nav.skipToContent': { en: 'Skip to main content', si: 'ප්‍රධාන අන්තර්ගතයට යන්න' },
  'nav.openNavigation': { en: 'Open navigation', si: 'මෙනුව විවෘත කරන්න' },
  'nav.closeNavigation': { en: 'Close navigation', si: 'මෙනුව වසන්න' },
  'nav.teacher': { en: 'Teacher', si: 'ගුරුවරයා' },
  'nav.student': { en: 'Student', si: 'ශිෂ්‍යයා' },

  'footer.accessibilityStatement': { en: 'Accessibility statement', si: 'ප්‍රවේශ්‍යතා ප්‍රකාශය' },
  'footer.privacyPolicy': { en: 'Privacy policy', si: 'රහස්‍යතා ප්‍රතිපත්තිය' },
  'footer.termsOfService': { en: 'Terms of service', si: 'සේවා නියම' },
  'footer.helpCenter': { en: 'Help center', si: 'උදව් මධ්‍යස්ථානය' },
  'footer.copyright': {
    en: (year) => `© ${year} SinhaSpeech Accessibility.`,
    si: (year) => `© ${year} SinhaSpeech ප්‍රවේශ්‍යතාව.`,
  },

  'settings.eyebrow': { en: 'Make SinhaSpeech work for you', si: 'SinhaSpeech ඔබට ගැලපෙන ලෙස සකසන්න' },
  'settings.title': { en: 'Accessibility settings', si: 'ප්‍රවේශ්‍යතා සැකසුම්' },
  'settings.description': {
    en: 'These preferences are saved on this device and applied across the application.',
    si: 'මෙම මනාපයන් මෙම උපාංගයේ සුරැකී යෙදුම පුරාම යොදනු ලැබේ.',
  },
  'settings.languageTitle': { en: 'Display language', si: 'ප්‍රදර්ශන භාෂාව' },
  'settings.languageDescription': {
    en: 'Choose the language used throughout the SinhaSpeech interface.',
    si: 'SinhaSpeech අතුරු මුහුණත පුරා භාවිතා කරන භාෂාව තෝරන්න.',
  },
  'settings.readingPreferences': { en: 'Reading preferences', si: 'කියවීමේ මනාපයන්' },
  'settings.confirmPreferences': { en: 'Confirm preferences', si: 'මනාපයන් තහවුරු කරන්න' },
  'settings.autoSavedToast': {
    en: 'Your accessibility settings are saved automatically.',
    si: 'ඔබේ ප්‍රවේශ්‍යතා සැකසුම් ස්වයංක්‍රීයව සුරැකේ.',
  },

  'lang.english': { en: 'English', si: 'ඉංග්‍රීසි' },
  'lang.sinhala': { en: 'Sinhala', si: 'සිංහල' },
}

export function translate(key, language, ...args) {
  const entry = translations[key]
  if (!entry) {
    if (import.meta.env?.DEV) console.warn(`Missing translation key: ${key}`)
    return key
  }
  const value = entry[language] ?? entry.en
  return typeof value === 'function' ? value(...args) : value
}
