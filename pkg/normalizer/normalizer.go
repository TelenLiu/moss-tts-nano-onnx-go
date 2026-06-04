package normalizer

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	chinese "github.com/TelenLiu/WeTextProcessing-go/tn/chinese"
	english "github.com/TelenLiu/WeTextProcessing-go/tn/english"
)

var (
	zhNormalizer *chinese.Normalizer
	enNormalizer *english.Normalizer

	zhOnce sync.Once
	enOnce sync.Once

	zhReady bool
	enReady bool
)

var keepHyphenPlaceholder = "___KEEP_HYPHEN_BEFORE_ZH_WETEXT___"

var (
	zhCacheDir = filepath.Join("lib", "cache", "wetext_zh")
	enCacheDir = filepath.Join("lib", "cache", "wetext_en")
)

func initZh(progress ...func(stage string, current, total int)) {
	var pf func(stage string, current, total int)
	if len(progress) > 0 {
		pf = progress[0]
	}
	zhNormalizer = chinese.NewNormalizer(
		zhCacheDir,
		false,
		false, false, false, false, false, false,
		pf,
	)
	zhReady = true
}

func initEn(progress ...func(stage string, current, total int)) {
	var pf func(stage string, current, total int)
	if len(progress) > 0 {
		pf = progress[0]
	}
	enNormalizer = english.NewNormalizer(
		enCacheDir,
		false,
		pf,
	)
	enReady = true
}

type ProgressFunc func(msg string, percent float64)

func EnsureInitialized() {
	go zhOnce.Do(func() { initZh() })
	go enOnce.Do(func() { initEn() })
}

func EnsureInitializedSync(progress ProgressFunc) {
	if progress != nil {
		progress("正在初始化文本归一化引擎...", 8)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	zhStart, zhEnd := 50.0, 74.0
	enStart, enEnd := 75.0, 99.0
	go func() {
		defer wg.Done()
		zhOnce.Do(func() {
			initZh(func(stage string, current, total int) {
				if progress != nil {
					pct := zhStart + float64(current)/float64(total)*(zhEnd-zhStart)
					progress(fmt.Sprintf("中文: %s (%d/%d)", stage, current, total), pct)
				}
			})
			if progress != nil {
				progress("中文文本归一化引擎就绪", zhEnd)
			}
		})
	}()
	go func() {
		defer wg.Done()
		enOnce.Do(func() {
			initEn(func(stage string, current, total int) {
				if progress != nil {
					pct := enStart + float64(current)/float64(total)*(enEnd-enStart)
					progress(fmt.Sprintf("英文: %s (%d/%d)", stage, current, total), pct)
				}
			})
			if progress != nil {
				progress("英文文本归一化引擎就绪", enEnd)
			}
		})
	}()
	wg.Wait()
}

func verifyCacheFiles(cacheDir, prefix string) error {
	required := []string{
		prefix + "_base_vsig.fst",
		prefix + "_base_char.fst",
		prefix + "_base_sigma.fst",
		prefix + "_base_not_quote.fst",
		prefix + "_base_not_space.fst",
		prefix + "_base_to_lower.fst",
		prefix + "_base_to_upper.fst",
		prefix + "_base_insert_space.fst",
		prefix + "_base_delete_space.fst",
		prefix + "_base_delete_extra_space.fst",
		prefix + "_base_delete_zero_one_space.fst",
		prefix + "_tagger.fst",
		prefix + "_verbalizer.fst",
	}
	var missing []string
	for _, name := range required {
		p := filepath.Join(cacheDir, name)
		if _, err := os.Stat(p); err != nil {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("缺少 %d 个缓存文件: %v", len(missing), missing)
	}
	return nil
}

func BuildCache() error {
	log.Printf("[normalizer] 开始构建 FST 缓存 (中文TN + 英文TN)")

	buildLang := func(name, cacheDir, prefix string, fn func(func(string, int, int))) error {
		log.Printf("[normalizer] %s: 开始构建", name)
		t0 := time.Now()
		fn(func(stage string, current, total int) {
			log.Printf("[normalizer] %s: %s (%d/%d)", name, stage, current, total)
		})
		elapsed := time.Since(t0).Round(time.Second)
		if err := verifyCacheFiles(cacheDir, prefix); err != nil {
			log.Printf("[normalizer] %s: 完成但缓存文件验证失败: %v", name, err)
			return fmt.Errorf("%s 缓存文件不完整: %v", name, err)
		}
		log.Printf("[normalizer] %s: 完成 (%v), 缓存文件已就绪", name, elapsed)
		return nil
	}

	if err := buildLang("中文TN", zhCacheDir, "zh_tn", func(p func(string, int, int)) {
		zhNormalizer = chinese.NewNormalizer(
			zhCacheDir, true,
			false, false, false, false, false, false,
			p,
		)
		zhReady = true
	}); err != nil {
		return fmt.Errorf("中文TN归一化缓存构建失败: %v", err)
	}

	if err := buildLang("英文TN", enCacheDir, "en_tn", func(p func(string, int, int)) {
		enNormalizer = english.NewNormalizer(
			enCacheDir, true,
			p,
		)
		enReady = true
	}); err != nil {
		return fmt.Errorf("英文TN归一化缓存构建失败: %v", err)
	}

	log.Println("[normalizer] FST 缓存构建完成")
	return nil
}

func IsZhReady() bool {
	return zhReady
}

func IsEnReady() bool {
	return enReady
}

func IsReady() bool {
	return zhReady && enReady
}

func ContainsCJK(text string) bool {
	for _, ch := range text {
		if (ch >= '\u3400' && ch <= '\u4dbf') ||
			(ch >= '\u4e00' && ch <= '\u9fff') ||
			(ch >= '\u3040' && ch <= '\u30ff') ||
			(ch >= '\uac00' && ch <= '\ud7af') {
			return true
		}
	}
	return false
}

func ResolveNormalizationLanguage(text string) string {
	if cjkRegex.MatchString(text) {
		return "zh"
	}
	hasLatin := false
	for _, ch := range text {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') {
			hasLatin = true
			break
		}
	}
	if hasLatin {
		return "en"
	}
	return "zh"
}

func RewriteHyphensBeforeZhWeText(text string) string {
	if !strings.Contains(text, "-") {
		return text
	}
	rewritten := text
	reStartNeg := regexp.MustCompile(`(^\s*)-\s*(\d)`)
	rewritten = reStartNeg.ReplaceAllString(rewritten, "${1}"+keepHyphenPlaceholder+"${2}")
	reAfterDelim := regexp.MustCompile(`([=:+*/,(，:：；;（【\[{])\s*-\s*(\d)`)
	rewritten = reAfterDelim.ReplaceAllString(rewritten, "${1}"+keepHyphenPlaceholder+"${2}")
	reZhNeg := regexp.MustCompile(`([\x{3400}-\x{9fff}])\s*-\s*(\d)`)
	rewritten = reZhNeg.ReplaceAllString(rewritten, "${1}"+keepHyphenPlaceholder+"${2}")
	reNumRange := regexp.MustCompile(`(\d)\s*-\s*(\d)`)
	rewritten = reNumRange.ReplaceAllString(rewritten, "${1}"+keepHyphenPlaceholder+"${2}")
	reZhPhrase := regexp.MustCompile(`([\x{3400}-\x{9fff}])\s*-\s*([\x{3400}-\x{9fff}])`)
	rewritten = reZhPhrase.ReplaceAllString(rewritten, "$1，$2")
	reOther := regexp.MustCompile(`([^\s-])\s*-\s*([^\s-])`)
	rewritten = reOther.ReplaceAllString(rewritten, "$1 $2")
	multiSpace := regexp.MustCompile(` {2,}`)
	rewritten = strings.TrimSpace(multiSpace.ReplaceAllString(rewritten, " "))
	return strings.ReplaceAll(rewritten, keepHyphenPlaceholder, "-")
}

func NormalizeTTSText(text string) string {
	if text == "" {
		return text
	}
	zhOnce.Do(func() { initZh() })
	normalized := zhNormalizer.Normalize(text)
	normalized = strings.TrimSpace(normalized)
	return normalized
}

func NormalizeWithWeText(text string, language string) string {
	if text == "" {
		return text
	}
	var result string
	switch language {
	case "en":
		enOnce.Do(func() { initEn() })
		result = enNormalizer.Normalize(text)
	default:
		zhOnce.Do(func() { initZh() })
		result = zhNormalizer.Normalize(text)
	}
	return strings.TrimSpace(result)
}

func PrepareTTSText(text string, enableRobust bool, enableWeText bool) string {
	if text == "" {
		return text
	}
	current := text
	if enableRobust {
		current = NormalizeRobust(current)
	}
	if enableWeText {
		lang := ResolveNormalizationLanguage(current)
		if lang == "zh" {
			current = RewriteHyphensBeforeZhWeText(current)
		}
		current = NormalizeWithWeText(current, lang)
	}
	if enableRobust {
		current = NormalizeRobust(current)
	}
	return current
}
