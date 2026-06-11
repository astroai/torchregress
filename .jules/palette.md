## 2024-06-11 - Use explicit HTML for image badges
**Learning:** Using standard Markdown for image links (like CI or PyPI badges) lacks accessibility because standard Markdown formatting does not allow specifying an `aria-label` attribute on the resulting `<a>` tag.
**Action:** When adding or maintaining image badges in documentation (like READMEs), use explicit HTML `<a>` and `<img>` tags to ensure screen readers receive proper context through `aria-label`.
