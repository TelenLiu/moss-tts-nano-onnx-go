package normalizer

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

var cjkRe = regexp.MustCompile(`[\x{3400}-\x{4dbf}\x{4e00}-\x{9fff}\x{3040}-\x{30ff}]`)
var protRe = regexp.MustCompile(`___PROT\d+___`)
var urlRe = regexp.MustCompile(`https?://[^\s\x{3000}，。！？；、）】》〉」』]+`)
var mentionRe = regexp.MustCompile(`@[A-Za-z0-9_]{1,32}`)
var hashtagRe = regexp.MustCompile(`#[^\s#]+`)
var dotTokenRe = regexp.MustCompile(`\.[A-Za-z0-9][A-Za-z0-9._-]*`)
var filelikeRe = regexp.MustCompile(`[A-Za-z0-9][A-Za-z0-9._/+:-]*[./+:-][A-Za-z0-9._/+:-]*`)
var zeroWidthRe = regexp.MustCompile(`[\x{200b}-\x{200d}\x{feff}]`)
var flowArrowRe = regexp.MustCompile(`\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*`)
var markdownLinkRe = regexp.MustCompile(`\[([^\[\]]+?)\]\((https?://[^)\s]+)\)`)
var headingRe = regexp.MustCompile(`^#{1,6}\s+`)
var blockquoteRe = regexp.MustCompile(`^>\s+`)
var unorderedListRe = regexp.MustCompile(`^[-*+]\s+`)
var orderedListRe = regexp.MustCompile(`^\d+[.)]\s+`)
var whitespaceRe = regexp.MustCompile(`[ \t\r\f\v]+`)
var multiSpaceRe = regexp.MustCompile(` {2,}`)
var ellipsisRe = regexp.MustCompile(`(?:\.{3,}|…{2,}|……+)`)
var multiPeriodRe = regexp.MustCompile(`[。．]{2,}`)
var multiCommaRe = regexp.MustCompile(`[，,]{2,}`)
var multiExclRe = regexp.MustCompile(`[!！]{2,}`)
var multiQuesRe = regexp.MustCompile(`[?？]{2,}`)
var mixedQERe = regexp.MustCompile(`[!?！？]{2,}`)
var longDashRe = regexp.MustCompile(`\s*(?:—|–|―|-){2,}\s*`)

var trailingClosers = map[rune]bool{
	'"': true, '\'': true, ')': true, ']': true, '}': true,
	'）': true, '】': true, '》': true, '〉': true, '」': true, '』': true,
	'\u201d': true, '\u2019': true,
}

func isCJK(ch rune) bool {
	return (ch >= '\u3400' && ch <= '\u4dbf') || (ch >= '\u4e00' && ch <= '\u9fff') || (ch >= '\u3040' && ch <= '\u30ff')
}

func isPunctuation(ch rune) bool {
	return unicode.IsPunct(ch) || unicode.IsSymbol(ch)
}

func ContainsCJK(text string) bool {
	return cjkRe.MatchString(text)
}

func NormalizeTTSText(text string) string {
	text = baseCleanup(text)
	text = normalizeMarkdownAndLines(text)
	text = normalizeFlowArrows(text)
	text, protected := protectSpans(text)
	text = normalizeVisibleUnderscores(text)
	text = normalizeSpaces(text)
	text = normalizeStructuralPunctuation(text)
	text = normalizeRepeatedPunctuation(text)
	text = normalizeSpaces(text)
	text = restoreSpans(text, protected)
	text = strings.TrimSpace(text)
	return ensureTerminalPunctuationByLine(text)
}

func baseCleanup(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.ReplaceAll(text, "\u3000", " ")
	text = zeroWidthRe.ReplaceAllString(text, "")
	var cleaned strings.Builder
	for _, ch := range text {
		if ch == '\n' || ch == '\t' || ch == ' ' || !unicode.IsControl(ch) {
			cleaned.WriteRune(ch)
		}
	}
	return cleaned.String()
}

func normalizeMarkdownAndLines(text string) string {
	text = markdownLinkRe.ReplaceAllString(text, "$1 $2")
	lines := strings.Split(text, "\n")
	var processed []string
	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		line = headingRe.ReplaceAllString(line, "")
		line = blockquoteRe.ReplaceAllString(line, "")
		line = unorderedListRe.ReplaceAllString(line, "")
		line = orderedListRe.ReplaceAllString(line, "")
		processed = append(processed, line)
	}
	if len(processed) == 0 {
		return ""
	}
	var merged []string
	merged = append(merged, processed[0])
	for i := 1; i < len(processed); i++ {
		merged[len(merged)-1] = ensureTerminalPunctuation(merged[len(merged)-1])
		merged = append(merged, processed[i])
	}
	return strings.Join(merged, "")
}

func protectSpans(text string) (string, []string) {
	var protected []string
	replacer := func(match string) string {
		idx := len(protected)
		protected = append(protected, match)
		return fmt.Sprintf("___PROT%d___", idx)
	}
	patterns := []*regexp.Regexp{urlRe, mentionRe, hashtagRe, dotTokenRe, filelikeRe}
	for _, pat := range patterns {
		text = pat.ReplaceAllStringFunc(text, replacer)
	}
	return text, protected
}

func restoreSpans(text string, protected []string) string {
	for idx, original := range protected {
		text = strings.ReplaceAll(text, fmt.Sprintf("___PROT%d___", idx), original)
	}
	return text
}

func normalizeVisibleUnderscores(text string) string {
	matches := protRe.FindAllStringIndex(text, -1)
	if len(matches) == 0 {
		return strings.ReplaceAll(text, "_", " ")
	}
	result := ""
	lastEnd := 0
	for _, loc := range matches {
		chunk := text[lastEnd:loc[0]]
		result += strings.ReplaceAll(chunk, "_", " ")
		result += text[loc[0]:loc[1]]
		lastEnd = loc[1]
	}
	if lastEnd < len(text) {
		result += strings.ReplaceAll(text[lastEnd:], "_", " ")
	}
	return result
}

func normalizeFlowArrows(text string) string {
	return flowArrowRe.ReplaceAllString(text, "，")
}

func normalizeSpaces(text string) string {
	text = whitespaceRe.ReplaceAllString(text, " ")
	text = cjkSpaceIntra(text)
	text = multiSpaceRe.ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

func cjkSpaceIntra(text string) string {
	runes := []rune(text)
	var result []rune
	for i, ch := range runes {
		if ch == ' ' {
			prevIsCJK := i > 0 && isCJK(runes[i-1])
			nextIsCJK := i < len(runes)-1 && isCJK(runes[i+1])
			prevIsDigit := i > 0 && runes[i-1] >= '0' && runes[i-1] <= '9'
			nextIsDigit := i < len(runes)-1 && runes[i+1] >= '0' && runes[i+1] <= '9'
			if prevIsCJK && nextIsCJK {
				continue
			}
			if prevIsCJK && nextIsDigit {
				continue
			}
			if prevIsDigit && nextIsCJK {
				continue
			}
		}
		result = append(result, ch)
	}
	return string(result)
}

func normalizeStructuralPunctuation(text string) string {
	text = normalizeFlowArrows(text)
	text = longDashRe.ReplaceAllString(text, "。")
	return text
}

func normalizeRepeatedPunctuation(text string) string {
	text = ellipsisRe.ReplaceAllString(text, "。")
	text = multiPeriodRe.ReplaceAllString(text, "。")
	text = multiCommaRe.ReplaceAllString(text, "，")
	text = multiExclRe.ReplaceAllString(text, "！")
	text = multiQuesRe.ReplaceAllString(text, "？")
	text = mixedQERe.ReplaceAllStringFunc(text, func(match string) string {
		hasQ := strings.ContainsAny(match, "?？")
		hasE := strings.ContainsAny(match, "!！")
		if hasQ && hasE {
			return "？！"
		}
		if hasQ {
			return "？"
		}
		return "！"
	})
	return text
}

func ensureTerminalPunctuation(text string) string {
	if text == "" {
		return text
	}
	runes := []rune(text)
	index := len(runes) - 1
	for index >= 0 && (runes[index] == ' ' || runes[index] == '\t') {
		index--
	}
	for index >= 0 && trailingClosers[runes[index]] {
		index--
	}
	if index >= 0 && isPunctuation(runes[index]) {
		return text
	}
	return text + "。"
}

func ensureTerminalPunctuationByLine(text string) string {
	if text == "" {
		return text
	}
	lines := strings.Split(text, "\n")
	var normalized []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			normalized = append(normalized, ensureTerminalPunctuation(trimmed))
		}
	}
	return strings.TrimSpace(strings.Join(normalized, "\n"))
}
