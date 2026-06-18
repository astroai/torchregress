## 2024-06-11 - Use explicit HTML for image badges
**Learning:** Using standard Markdown for image links (like CI or PyPI badges) lacks accessibility because standard Markdown formatting does not allow specifying an `aria-label` attribute on the resulting `<a>` tag.
**Action:** When adding or maintaining image badges in documentation (like READMEs), use explicit HTML `<a>` and `<img>` tags to ensure screen readers receive proper context through `aria-label`.
## 2025-02-12 - Adding Focus-Within for Grid Cards
**Learning:** Adding `:hover` effects to interactive components without providing equivalent focus states (like `:focus-within`) creates an inaccessible experience for keyboard users.
**Action:** Always include `:focus-within` when applying hover effects (like border color and box-shadow changes) on container elements that have interactive children (such as links in grid cards) to ensure keyboard parity.
