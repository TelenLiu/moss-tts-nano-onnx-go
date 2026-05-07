package normalizer

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

var cjkRe = regexp.MustCompile(`[\x{3400}-\x{4dbf}\x{4e00}-\x{9fff}\x{3040}-\x{30ff}\x{ac00}-\x{d7af}]`)
var protRe = regexp.MustCompile(`___PROT\d+___`)
var urlRe = regexp.MustCompile(`https?://[^\s\x{3000}，。！？；、）】》〉」』]+`)
var emailRe = regexp.MustCompile(`[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`)
var mentionRe = regexp.MustCompile(`@[A-Za-z0-9_]{1,32}`)
var redditRe = regexp.MustCompile(`(?:u|r)/[A-Za-z0-9_]+`)
var hashtagRe = regexp.MustCompile(`#[^\s#]+`)
var dotTokenRe = regexp.MustCompile(`\.[A-Za-z0-9][A-Za-z0-9._-]*`)
var filelikeRe = regexp.MustCompile(`[A-Za-z0-9][A-Za-z0-9._/+:\-]*[./+:\-][A-Za-z0-9._/+:\-]*`)
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

var cnPunctAfterRe = regexp.MustCompile(`\s+([，。！？；：、”’」』】）》])`)
var cnPunctBeforeRe = regexp.MustCompile(`([（【「『《“‘])\s+`)
var cnPunctStripRe = regexp.MustCompile(`([，。！？；：、])\s*`)
var asciiPunctRe = regexp.MustCompile(`\s+([,.;!?])`)

var squareBracketRe = regexp.MustCompile(`\[\s*([^\[\]]+?)\s*\]`)
var curlyBracketRe = regexp.MustCompile(`\{\s*([^{}]+?)\s*\}`)
var cnBracketRe = regexp.MustCompile(`[【〖『「]\s*([^】〗』」]+?)\s*[】〗』」]`)
var guillemetTitleRe = regexp.MustCompile(`《([^》]+)》`)

var latinishRe = regexp.MustCompile(`(?:___PROT\d+___|[A-Za-z0-9][A-Za-z0-9._/+:\-]*)`)

var trailingClosers = map[rune]bool{
	'"': true, '\'': true, ')': true, ']': true, '}': true,
	'）': true, '】': true, '》': true, '〉': true, '」': true, '』': true,
	'\u201d': true, '\u2019': true,
}

func convertFullwidthToHalfwidth(ch rune) (rune, bool) {
	if ch >= '\uFF01' && ch <= '\uFF5E' {
		return ch - 0xFEE0, true
	}
	switch ch {
	case '\uFFE0':
		return '¢', true
	case '\uFFE1':
		return '£', true
	case '\uFFE2':
		return '¬', true
	case '\uFFE3':
		return '¯', true
	case '\uFFE4':
		return '¦', true
	case '\uFFE5':
		return '¥', true
	case '\uFFE6':
		return '₩', true
	case '\u3000':
		return ' ', true
	case '。':
		return '\u0000', false
	case '、':
		return '\u0000', false
	}
	if ch >= '\uFE30' && ch <= '\uFE6B' {
		switch ch {
		case '\uFE30':
			return '\u2025', true
		case '\uFE31':
			return '\u2014', true
		case '\uFE32':
			return '\u2013', true
		case '\uFE33':
			return '_', true
		case '\uFE34':
			return '_', true
		case '\uFE35':
			return '(', true
		case '\uFE36':
			return ')', true
		case '\uFE37':
			return '{', true
		case '\uFE38':
			return '}', true
		case '\uFE39':
			return '(', true
		case '\uFE3A':
			return ')', true
		case '\uFE3B':
			return '{', true
		case '\uFE3C':
			return '}', true
		case '\uFE3D':
			return '[', true
		case '\uFE3E':
			return ']', true
		case '\uFE3F':
			return '(', true
		case '\uFE40':
			return ')', true
		case '\uFE41':
			return '{', true
		case '\uFE42':
			return '}', true
		case '\uFE43':
			return '(', true
		case '\uFE44':
			return ')', true
		case '\uFE49':
			return '\u203E', true
		case '\uFE4A':
			return '\u203E', true
		case '\uFE4B':
			return '\u203E', true
		case '\uFE4C':
			return '\u203E', true
		case '\uFE4D':
			return '_', true
		case '\uFE4E':
			return '_', true
		case '\uFE4F':
			return '_', true
		case '\uFE50':
			return ',', true
		case '\uFE51':
			return ',', true
		case '\uFE52':
			return '.', true
		case '\uFE54':
			return ';', true
		case '\uFE55':
			return ':', true
		case '\uFE56':
			return '?', true
		case '\uFE57':
			return '!', true
		case '\uFE58':
			return '-', true
		case '\uFE59':
			return '(', true
		case '\uFE5A':
			return ')', true
		case '\uFE5B':
			return '{', true
		case '\uFE5C':
			return '}', true
		case '\uFE5D':
			return '[', true
		case '\uFE5E':
			return ']', true
		case '\uFE5F':
			return '#', true
		case '\uFE60':
			return '&', true
		case '\uFE61':
			return '*', true
		case '\uFE62':
			return '+', true
		case '\uFE63':
			return '-', true
		case '\uFE64':
			return '<', true
		case '\uFE65':
			return '>', true
		case '\uFE66':
			return '=', true
		case '\uFE68':
			return '\\', true
		case '\uFE69':
			return '$', true
		case '\uFE6A':
			return '%', true
		case '\uFE6B':
			return '@', true
		}
	}
	return 0, false
}

func normalizeFullwidthPunctuation(text string) string {
	var result strings.Builder
	for _, ch := range text {
		if half, ok := convertFullwidthToHalfwidth(ch); ok {
			result.WriteRune(half)
		} else if ch == '。' || ch == '、' {
			result.WriteRune(ch)
		} else {
			result.WriteRune(ch)
		}
	}
	return result.String()
}

func isCJK(ch rune) bool {
	return (ch >= '\u3400' && ch <= '\u4dbf') || (ch >= '\u4e00' && ch <= '\u9fff') || (ch >= '\u3040' && ch <= '\u30ff') || (ch >= '\uac00' && ch <= '\ud7af')
}

func isPunctuation(ch rune) bool {
	return unicode.IsPunct(ch)
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
	text = normalizeFullwidthPunctuation(text)
	text = normalizeSpecialSymbols(text)
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
	patterns := []*regexp.Regexp{urlRe, emailRe, mentionRe, redditRe, hashtagRe, dotTokenRe, filelikeRe}
	for _, pat := range patterns {
		text = pat.ReplaceAllStringFunc(text, replacer)
	}
	return text, protected
}

func restoreSpans(text string, protected []string) string {
	for idx := len(protected) - 1; idx >= 0; idx-- {
		text = strings.ReplaceAll(text, fmt.Sprintf("___PROT%d___", idx), protected[idx])
	}
	return text
}

func normalizeVisibleUnderscores(text string) string {
	matches := protRe.FindAllStringIndex(text, -1)
	if len(matches) == 0 {
		return strings.ReplaceAll(text, "_", " ")
	}
	var result strings.Builder
	lastEnd := 0
	for _, loc := range matches {
		result.WriteString(strings.ReplaceAll(text[lastEnd:loc[0]], "_", " "))
		result.WriteString(text[loc[0]:loc[1]])
		lastEnd = loc[1]
	}
	if lastEnd < len(text) {
		result.WriteString(strings.ReplaceAll(text[lastEnd:], "_", " "))
	}
	return result.String()
}

func normalizeFlowArrows(text string) string {
	return flowArrowRe.ReplaceAllString(text, "，")
}

func normalizeSpaces(text string) string {
	text = whitespaceRe.ReplaceAllString(text, " ")

	text = cnCJKIntraSpace(text)
	text = cnCJKDigitSpace(text)

	text = cnPunctAfterRe.ReplaceAllString(text, "$1")
	text = cnPunctBeforeRe.ReplaceAllString(text, "$1")
	text = cnPunctStripRe.ReplaceAllString(text, "$1")
	text = asciiPunctRe.ReplaceAllString(text, "$1")

	text = multiSpaceRe.ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

func cnCJKIntraSpace(text string) string {
	return cjkRe.ReplaceAllStringFunc(text, func(match string) string {
		return match
	})
}

func cnCJKDigitSpace(text string) string {
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

func cnCJKLatinBoundarySpace(text string) string {
	runes := []rune(text)
	var result strings.Builder
	i := 0
	for i < len(runes) {
		ch := runes[i]
		if isCJK(ch) {
			result.WriteRune(ch)
			nextStart := i + 1
			for nextStart < len(runes) && runes[nextStart] == ' ' {
				nextStart++
			}
			if nextStart < len(runes) {
				if isLatinishAt(runes, nextStart) {
					result.WriteRune(' ')
				}
			}
			i++
			continue
		}
		if isLatinishAt(runes, i) {
			latinEnd := scanLatinish(runes, i)
			result.WriteString(string(runes[i:latinEnd]))
			i = latinEnd
			for i < len(runes) && runes[i] == ' ' {
				i++
			}
			if i < len(runes) && isCJK(runes[i]) {
				result.WriteRune(' ')
			}
			continue
		}
		result.WriteRune(ch)
		i++
	}
	return result.String()
}

func isLatinishAt(runes []rune, start int) bool {
	if start >= len(runes) {
		return false
	}
	ch := runes[start]
	return ch == '_' || ch == '.' || ch == '/' || ch == '+' || ch == ':' || ch == '-' || (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')
}

func scanLatinish(runes []rune, start int) int {
	end := start
	for end < len(runes) {
		ch := runes[end]
		if isCJK(ch) || unicode.Is(unicode.Scripts["Han"], ch) {
			break
		}
		if ch == ' ' || (!isLatinChar(ch) && !isLatinSpecial(ch)) {
			if end == start {
				end++
			}
			break
		}
		end++
	}
	return end
}

func isLatinChar(ch rune) bool {
	return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')
}

func isLatinSpecial(ch rune) bool {
	return ch == '_' || ch == '.' || ch == '/' || ch == '+' || ch == ':' || ch == '-'
}

func normalizeStructuralPunctuation(text string) string {
	text = squareBracketRe.ReplaceAllString(text, `"$1"`)
	text = curlyBracketRe.ReplaceAllString(text, `"$1"`)
	text = cnBracketRe.ReplaceAllString(text, `"$1"`)
	text = normalizeGuillemetTitles(text)
	text = normalizeFlowArrows(text)
	text = longDashRe.ReplaceAllString(text, "。")
	return text
}

func normalizeGuillemetTitles(text string) string {
	runes := []rune(text)
	var result []rune
	i := 0
	for i < len(runes) {
		if runes[i] == '《' {
			start := i
			end := -1
			for j := i + 1; j < len(runes); j++ {
				if runes[j] == '》' {
					end = j
					break
				}
			}
			if end > 0 {
				after := end + 1
				for after < len(runes) && (runes[after] == ' ' || runes[after] == '\t') {
					after++
				}
				isAfterSentencePunct := false
				if after < len(runes) {
					ch := runes[after]
					isAfterSentencePunct = (ch == '。' || ch == '！' || ch == '？' || ch == '!' || ch == '?' || ch == '；' || ch == ';' || ch == '，' || ch == ',')
				}
				isAtEnd := after >= len(runes)
				hasLongDashAfter := false
				if after < len(runes) {
					dashCount := 0
					for k := after; k < len(runes) && (runes[k] == '—' || runes[k] == '–' || runes[k] == '―' || runes[k] == '-'); k++ {
						dashCount++
					}
					hasLongDashAfter = dashCount >= 2
				}
				if isAtEnd || isAfterSentencePunct || hasLongDashAfter {
					prev := start - 1
					for prev >= 0 && runes[prev] == ' ' {
						prev--
					}
					hasPrefixPunct := false
					if prev >= 0 {
						ch := runes[prev]
						hasPrefixPunct = (ch == '。' || ch == '！' || ch == '？' || ch == '!' || ch == '?' || ch == '；' || ch == ';')
					}
					isAtStart := prev < 0
					if isAtStart || hasPrefixPunct {
						result = append(result, runes[start+1:end]...)
						i = end + 1
						continue
					}
				}
			}
		}
		result = append(result, runes[i])
		i++
	}
	return string(result)
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

var specialSymbolReplacer = strings.NewReplacer(
	"&", "和",
	"*", "",
	"|", "",
	"#", "号",
	"^", "",
	"<", "",
	">", "",
	"=", "等于",
	"+", "加",
	"%", "百分之",
)

var slashSepRe = regexp.MustCompile(`([\p{Han}]+)/([\p{Han}]+)`)
var backslashRe = regexp.MustCompile(`\\`)
var tildeNumRe = regexp.MustCompile(`~(\d)`)
var tildeOtherRe = regexp.MustCompile(`~`)
var atCJKRe = regexp.MustCompile(`([\p{Han}])@([\p{Han}])`)
var atCJKLeftRe = regexp.MustCompile(`([\p{Han}])@`)
var atCJKRightRe = regexp.MustCompile(`@([\p{Han}])`)

func normalizeSpecialSymbols(text string) string {
	text = backslashRe.ReplaceAllStringFunc(text, func(m string) string {
		return ""
	})
	text = slashSepRe.ReplaceAllStringFunc(text, func(m string) string {
		sub := slashSepRe.FindStringSubmatch(m)
		if len(sub) == 3 {
			return sub[1] + "和" + sub[2]
		}
		return m
	})
	text = tildeNumRe.ReplaceAllStringFunc(text, func(m string) string {
		return "到" + m[1:]
	})
	text = tildeOtherRe.ReplaceAllStringFunc(text, func(m string) string {
		return "至"
	})
	text = atCJKRe.ReplaceAllStringFunc(text, func(m string) string {
		sub := atCJKRe.FindStringSubmatch(m)
		if len(sub) == 3 {
			return sub[1] + "在" + sub[2]
		}
		return m
	})
	text = atCJKLeftRe.ReplaceAllStringFunc(text, func(m string) string {
		sub := atCJKLeftRe.FindStringSubmatch(m)
		if len(sub) >= 2 {
			return sub[1] + "在"
		}
		return m
	})
	text = atCJKRightRe.ReplaceAllStringFunc(text, func(m string) string {
		sub := atCJKRightRe.FindStringSubmatch(m)
		if len(sub) == 2 {
			return "在" + sub[1]
		}
		return m
	})
	text = specialSymbolReplacer.Replace(text)
	return text
}
