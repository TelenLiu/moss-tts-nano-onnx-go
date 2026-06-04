package normalizer

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

var cjkStr = "[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]"
var cjkRegex = regexp.MustCompile(cjkStr)
var cjkRunes = "\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"

var protRegex = regexp.MustCompile(`___PROT\d+___`)

var urlRegex = regexp.MustCompile(`https?://[^\s` + "\u3000" + `，。！？；、）】》〉」』]+`)

var hasLetter = regexp.MustCompile(`[A-Za-z]`)
var hasSpecial = regexp.MustCompile(`[./+:-]`)

var zeroWidthRegex = regexp.MustCompile("[\u200b-\u200d\ufeff]")

var markdownLinkRegex = regexp.MustCompile(`\[([^\[\]]+?)\]\((https?://[^)\s]+)\)`)

var flowArrowRegex = regexp.MustCompile(`\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*`)

var structuralBracket1 = regexp.MustCompile(`\[\s*([^\[\]]+?)\s*\]`)
var structuralBracket2 = regexp.MustCompile(`\{\s*([^{}]+?)\s*\}`)
var structuralBracket3 = regexp.MustCompile(`[【〖『「]\s*([^】〗』」]+?)\s*[】〗』」]`)

var longDashRegex = regexp.MustCompile(`\s*(?:—|–|―|-){2,}\s*`)

var repeatedEllipsis = regexp.MustCompile(`(?:\.{3,}|…{2,}|……+)`)
var repeatedPeriod = regexp.MustCompile(`[。．]{2,}`)
var repeatedComma = regexp.MustCompile(`[，,]{2,}`)
var repeatedExcl = regexp.MustCompile(`[!！]{2,}`)
var repeatedQues = regexp.MustCompile(`[?？]{2,}`)
var mixedQE = regexp.MustCompile(`[!?！？]{2,}`)

var headingRegex = regexp.MustCompile(`^#{1,6}\s+`)
var quoteRegex = regexp.MustCompile(`^>\s+`)
var unorderedListRegex = regexp.MustCompile(`^[-*+]\s+`)
var orderedListRegex = regexp.MustCompile(`^\d+[.)]\s+`)

var spaceRegex = regexp.MustCompile(`[ \t\r\f\v]+`)
var multiSpace = regexp.MustCompile(` {2,}`)

var trailingClosers = map[rune]bool{
	'"': true, '\'': true,
	')': true, ']': true, '}': true,
	'）': true, '】': true, '》': true, '〉': true, '」': true, '』': true, '”': true, '’': true,
}

func NormalizeRobust(text string) string {
	if text == "" {
		return text
	}
	result := baseCleanup(text)
	result = normalizeMarkdownAndLines(result)
	result = normalizeFlowArrows(result)
	result, protected := protectSpans(result)
	result = normalizeVisibleUnderscores(result)
	result = normalizeSpaces(result)
	result = normalizeStructuralPunctuation(result)
	result = normalizeRepeatedPunctuation(result)
	result = normalizeSpaces(result)
	result = restoreSpans(result, protected)
	result = strings.TrimSpace(result)
	return ensureTerminalPunctuationByLine(result)
}

func baseCleanup(text string) string {
	result := strings.ReplaceAll(text, "\r\n", "\n")
	result = strings.ReplaceAll(result, "\r", "\n")
	result = strings.ReplaceAll(result, "\u3000", " ")
	result = zeroWidthRegex.ReplaceAllString(result, "")

	var cleaned strings.Builder
	for _, ch := range result {
		cat := unicode.IsControl(ch)
		if ch == '\n' || ch == '\t' || ch == ' ' || !cat {
			cleaned.WriteRune(ch)
		}
	}
	return cleaned.String()
}

func isWordByte(ch byte) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_'
}

func protectSpans(text string) (string, []string) {
	var protected []string

	protectFn := func(match string) string {
		idx := len(protected)
		protected = append(protected, match)
		return fmt.Sprintf("___PROT%d___", idx)
	}

	text = urlRegex.ReplaceAllStringFunc(text, protectFn)

	text = protectWithBoundary(text, &protected, `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`)

	text = protectWithBoundary(text, &protected, `@[A-Za-z0-9_]{1,32}`)

	text = protectWithBoundary(text, &protected, `(?:u|r)/[A-Za-z0-9_]+`)

	text = protectWithBoundary(text, &protected, `#[^\s#]+`)

	text = protectWithBoundary(text, &protected, `\.[A-Za-z0-9._-]*[A-Za-z0-9]`)

	text = protectFilelike(text, &protected)

	return text, protected
}

type protectPat struct {
	pattern *regexp.Regexp
	check   func(text string, start, end int, match string) bool
}

func protectWithBoundary(text string, protected *[]string, pattern string) string {
	re := regexp.MustCompile(pattern)
	var result strings.Builder
	lastEnd := 0
	locs := re.FindAllStringIndex(text, -1)
	for _, loc := range locs {
		if loc[0] > 0 && isWordByte(text[loc[0]-1]) {
			result.WriteString(text[lastEnd:loc[1]])
			lastEnd = loc[1]
			continue
		}
		if loc[1] < len(text) && isWordByte(text[loc[1]]) {
			result.WriteString(text[lastEnd:loc[1]])
			lastEnd = loc[1]
			continue
		}
		result.WriteString(text[lastEnd:loc[0]])
		idx := len(*protected)
		*protected = append(*protected, text[loc[0]:loc[1]])
		result.WriteString(fmt.Sprintf("___PROT%d___", idx))
		lastEnd = loc[1]
	}
	result.WriteString(text[lastEnd:])
	return result.String()
}

var filelikeInner = regexp.MustCompile(`[A-Za-z0-9][A-Za-z0-9._/+:-]*[A-Za-z0-9]`)

func protectFilelike(text string, protected *[]string) string {
	var result strings.Builder
	lastEnd := 0
	locs := filelikeInner.FindAllStringIndex(text, -1)
	for _, loc := range locs {
		match := text[loc[0]:loc[1]]

		if loc[0] > 0 && isWordByte(text[loc[0]-1]) {
			result.WriteString(text[lastEnd:loc[1]])
			lastEnd = loc[1]
			continue
		}
		if loc[1] < len(text) && isWordByte(text[loc[1]]) {
			result.WriteString(text[lastEnd:loc[1]])
			lastEnd = loc[1]
			continue
		}

		if !hasLetter.MatchString(match) || !hasSpecial.MatchString(match) {
			result.WriteString(text[lastEnd:loc[1]])
			lastEnd = loc[1]
			continue
		}

		result.WriteString(text[lastEnd:loc[0]])
		idx := len(*protected)
		*protected = append(*protected, match)
		result.WriteString(fmt.Sprintf("___PROT%d___", idx))
		lastEnd = loc[1]
	}
	result.WriteString(text[lastEnd:])
	return result.String()
}

func restoreSpans(text string, protected []string) string {
	result := text
	for i, original := range protected {
		placeholder := fmt.Sprintf("___PROT%d___", i)
		result = strings.ReplaceAll(result, placeholder, original)
	}
	return result
}

func normalizeVisibleUnderscores(text string) string {
	var result strings.Builder
	lastEnd := 0
	loc := protRegex.FindAllStringIndex(text, -1)
	for _, l := range loc {
		before := text[lastEnd:l[0]]
		result.WriteString(strings.ReplaceAll(before, "_", " "))
		result.WriteString(text[l[0]:l[1]])
		lastEnd = l[1]
	}
	if lastEnd < len(text) {
		result.WriteString(strings.ReplaceAll(text[lastEnd:], "_", " "))
	}
	return result.String()
}

func normalizeFlowArrows(text string) string {
	return flowArrowRegex.ReplaceAllString(text, "，")
}

func normalizeMarkdownAndLines(text string) string {
	result := markdownLinkRegex.ReplaceAllString(text, "$1 $2")

	lines := strings.Split(result, "\n")
	var processed []string
	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		line = headingRegex.ReplaceAllString(line, "")
		line = quoteRegex.ReplaceAllString(line, "")
		line = unorderedListRegex.ReplaceAllString(line, "")
		line = orderedListRegex.ReplaceAllString(line, "")
		processed = append(processed, line)
	}

	if len(processed) == 0 {
		return ""
	}

	merged := make([]string, 0, len(processed))
	merged = append(merged, processed[0])
	for i := 1; i < len(processed); i++ {
		prev := merged[len(merged)-1]
		merged[len(merged)-1] = ensureTerminalPunctuation(prev)
		merged = append(merged, processed[i])
	}
	return strings.Join(merged, "")
}

func isCJK(ch rune) bool {
	return (ch >= '\u3400' && ch <= '\u4dbf') ||
		(ch >= '\u4e00' && ch <= '\u9fff') ||
		(ch >= '\u3040' && ch <= '\u30ff')
}

func isLatinishChar(ch rune) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
		(ch >= '0' && ch <= '9') || ch == '.' || ch == '/' || ch == '+' ||
		ch == ':' || ch == '-' || ch == '_'
}

func hasLetterInSeq(runes []rune, i int) bool {
	if !isLatinishChar(runes[i]) {
		return false
	}
	for j := i; j >= 0; j-- {
		r := runes[j]
		if isCJK(r) || r == ' ' || r == '@' || r == '#' {
			break
		}
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' {
			return true
		}
	}
	return false
}

func normalizeSpaces(text string) string {
	result := spaceRegex.ReplaceAllString(text, " ")

	runes := []rune(result)

	var cjkRemoved strings.Builder
	for i := 0; i < len(runes); i++ {
		ch := runes[i]
		if isCJK(ch) && i+1 < len(runes) && runes[i+1] == ' ' && i+2 < len(runes) && isCJK(runes[i+2]) {
			cjkRemoved.WriteRune(ch)
			i++
			continue
		}
		if isCJK(ch) && i+1 < len(runes) && runes[i+1] == ' ' && i+2 < len(runes) && runes[i+2] >= '0' && runes[i+2] <= '9' {
			cjkRemoved.WriteRune(ch)
			i++
			continue
		}
		if ch >= '0' && ch <= '9' && i+1 < len(runes) && runes[i+1] == ' ' && i+2 < len(runes) && isCJK(runes[i+2]) {
			cjkRemoved.WriteRune(ch)
			i++
			continue
		}
		cjkRemoved.WriteRune(ch)
	}
	result = cjkRemoved.String()

	runes = []rune(result)

	var spaced strings.Builder
	for i := 0; i < len(runes); i++ {
		ch := runes[i]
		spaced.WriteRune(ch)

		if isCJK(ch) && i+1 < len(runes) {
			next := runes[i+1]
			if isLatinishChar(next) && hasLetterInSeq(runes, i+1) {
				spaced.WriteRune(' ')
			}
		} else if i+1 < len(runes) && isCJK(runes[i+1]) {
			if isLatinishChar(ch) && hasLetterInSeq(runes, i) {
				spaced.WriteRune(' ')
			}
		}
	}
	result = spaced.String()

	result = multiSpace.ReplaceAllString(result, " ")

	cjkPunctRe := regexp.MustCompile(`([` + cjkRunes + `])(___PROT\d+___)`)
	result = cjkPunctRe.ReplaceAllString(result, "$1 $2")
	protCjkRe := regexp.MustCompile(`(___PROT\d+___)([` + cjkRunes + `])`)
	result = protCjkRe.ReplaceAllString(result, "$1 $2")

	result = multiSpace.ReplaceAllString(result, " ")

	chPunctAfter := regexp.MustCompile(`\s+([，。！？；：、”’」』】）》])`)
	result = chPunctAfter.ReplaceAllString(result, "$1")
	chPunctBefore := regexp.MustCompile(`([（【「『《“‘])\s+`)
	result = chPunctBefore.ReplaceAllString(result, "$1")
	chPunctCompact := regexp.MustCompile(`([，。！？；：、])\s*`)
	result = chPunctCompact.ReplaceAllString(result, "$1")

	asciiPunct := regexp.MustCompile(`\s+([,.;!?])`)
	result = asciiPunct.ReplaceAllString(result, "$1")

	return strings.TrimSpace(multiSpace.ReplaceAllString(result, " "))
}

func normalizeStructuralPunctuation(text string) string {
	result := structuralBracket1.ReplaceAllString(text, `"$1"`)
	result = structuralBracket2.ReplaceAllString(result, `"$1"`)
	result = structuralBracket3.ReplaceAllString(result, `"$1"`)

	titleRe := regexp.MustCompile(`(^|[。！？!?；;]\s*)《([^》]+)》`)
	result = titleRe.ReplaceAllStringFunc(result, func(match string) string {
		submatch := titleRe.FindStringSubmatch(match)
		if len(submatch) < 3 {
			return match
		}
		prefix := submatch[1]
		content := submatch[2]

		idx := strings.Index(result, match)
		if idx < 0 {
			return match
		}
		afterPos := idx + len(match)
		rest := strings.TrimSpace(result[afterPos:])

		if rest == "" || strings.HasPrefix(rest, "___PROT") {
			return prefix + content
		}
		dashPrefixes := []string{"—", "–", "―", "--"}
		for _, d := range dashPrefixes {
			if strings.HasPrefix(rest, d) {
				return prefix + content
			}
		}
		afterPunct := regexp.MustCompile(`^[。！？!?；;，,].*`)
		if afterPunct.MatchString(rest) {
			return prefix + content
		}
		return match
	})

	result = normalizeFlowArrows(result)
	result = longDashRegex.ReplaceAllString(result, "。")
	return result
}

func normalizeRepeatedPunctuation(text string) string {
	result := repeatedEllipsis.ReplaceAllString(text, "。")
	result = repeatedPeriod.ReplaceAllString(result, "。")
	result = repeatedComma.ReplaceAllString(result, "，")
	result = repeatedExcl.ReplaceAllString(result, "！")
	result = repeatedQues.ReplaceAllString(result, "？")
	result = mixedQE.ReplaceAllStringFunc(result, func(s string) string {
		hasQ := false
		hasE := false
		for _, ch := range s {
			if ch == '?' || ch == '？' {
				hasQ = true
			}
			if ch == '!' || ch == '！' {
				hasE = true
			}
		}
		if hasQ && hasE {
			return "？！"
		}
		if hasQ {
			return "？"
		}
		return "！"
	})
	return result
}

func ensureTerminalPunctuation(text string) string {
	if text == "" {
		return text
	}
	runes := []rune(text)
	idx := len(runes) - 1
	for idx >= 0 && unicode.IsSpace(runes[idx]) {
		idx--
	}
	for idx >= 0 && trailingClosers[runes[idx]] {
		idx--
	}
	if idx >= 0 && unicode.IsPunct(runes[idx]) {
		return text
	}
	return text + "。"
}

func ensureTerminalPunctuationByLine(text string) string {
	if text == "" {
		return text
	}
	lines := strings.Split(text, "\n")
	var resultLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			resultLines = append(resultLines, "")
		} else {
			resultLines = append(resultLines, ensureTerminalPunctuation(trimmed))
		}
	}
	return strings.TrimSpace(strings.Join(resultLines, "\n"))
}