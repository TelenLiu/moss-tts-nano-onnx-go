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
	tn "github.com/TelenLiu/WeTextProcessing-go/tn"
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

type ProgressFunc func(msg string, percent float64)

type ProgressExFunc func(msg string, percent float64, current, total int, elapsed time.Duration, estimatedRemaining time.Duration)

func initZh(progressEx ProgressExFunc, progress ...func(stage string, current, total int)) {
	// Build extended progress callback for NewNormalizerEx
	var exFn tn.BuildProgressExFn
	if progressEx != nil {
		exFn = func(stage string, current, total int, elapsed time.Duration, estimatedRemaining time.Duration) {
			progressEx(fmt.Sprintf("中文: %s (%d/%d)", stage, current, total), 0, current, total, elapsed, estimatedRemaining)
		}
	}
	// Use NewNormalizerEx with exFn for time estimates.
	// Don't pass basic progress fn to avoid duplicate progress reporting via ReportProgress
	// (which calls both buildProgress and buildProgressEx).
	zhNormalizer = chinese.NewNormalizerEx(
		zhCacheDir,
		false,
		false, false, false, false, false, false,
		exFn,
	)
	zhReady = true
}

func initEn(progressEx ProgressExFunc, progress ...func(stage string, current, total int)) {
	var pf func(stage string, current, total int)
	if len(progress) > 0 {
		pf = progress[0]
	}
	var exFn tn.BuildProgressExFn
	if progressEx != nil {
		exFn = func(stage string, current, total int, elapsed time.Duration, estimatedRemaining time.Duration) {
			progressEx(fmt.Sprintf("英文: %s (%d/%d)", stage, current, total), 0, current, total, elapsed, estimatedRemaining)
		}
	}
	enNormalizer = english.NewNormalizerEx(
		enCacheDir,
		false,
		exFn,
		pf,
	)
	enReady = true
}

func EnsureInitialized() {
	go zhOnce.Do(func() { initZh(nil) })
	go enOnce.Do(func() { initEn(nil) })
}

func EnsureInitializedSync(progress ProgressFunc, progressEx ...ProgressExFunc) {
	var pex ProgressExFunc
	if len(progressEx) > 0 {
		pex = progressEx[0]
	}
	if progress != nil {
		progress("正在初始化文本归一化引擎...", 8)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	zhStart, zhEnd := 50.0, 74.0
	enStart, enEnd := 75.0, 97.0

	// Chinese normalizer has two phases:
	//   Phase 1: "构建Tagger-xxx" or "加载缓存-xxx" (11 steps)
	//   Phase 2: "优化-xxx" (10 steps)
	// Total: 21 steps, phase1 = 11/21, phase2 = 10/21
	// Note: "构建基座FST" and "完成" are sub-steps within phase 1, ignored for pct calculation
	zhProgressEx := func(msg string, pct float64, current, total int, elapsed time.Duration, estimatedRemaining time.Duration) {
		var step, totalSteps float64
		if strings.HasPrefix(msg, "中文: 构建Tagger-") || strings.HasPrefix(msg, "中文: 加载缓存-") {
			// Phase 1: steps 1..11
			step = float64(current)
			totalSteps = 21.0
		} else if strings.HasPrefix(msg, "中文: 优化-") {
			// Phase 2: steps 12..21
			step = 11.0 + float64(current)
			totalSteps = 21.0
		} else {
			// Sub-steps like "构建基座FST", "完成" - map to first portion of phase 1
			step = float64(current) * 11.0 / float64(total) * 0.5 // use half of phase 1
			totalSteps = 21.0
		}
		calculatedPct := zhStart + (step/totalSteps)*(zhEnd-zhStart)
		if pex != nil {
			pex(msg, calculatedPct, current, total, elapsed, estimatedRemaining)
		} else if progress != nil {
			progress(msg, calculatedPct)
		}
	}

	// English normalizer has two phases:
	//   Phase 1: "构建Tagger-xxx" or "加载缓存-xxx" (14 steps)
	//   Phase 2: "优化-xxx" (14 steps)
	// Total: 28 steps, phase1 = 14/28, phase2 = 14/28
	enProgressEx := func(msg string, pct float64, current, total int, elapsed time.Duration, estimatedRemaining time.Duration) {
		var step, totalSteps float64
		if strings.HasPrefix(msg, "英文: 构建Tagger-") || strings.HasPrefix(msg, "英文: 加载缓存-") {
			step = float64(current)
			totalSteps = 28.0
		} else if strings.HasPrefix(msg, "英文: 优化-") {
			step = 14.0 + float64(current)
			totalSteps = 28.0
		} else {
			step = float64(current) * 14.0 / float64(total) * 0.5
			totalSteps = 28.0
		}
		calculatedPct := enStart + (step/totalSteps)*(enEnd-enStart)
		if pex != nil {
			pex(msg, calculatedPct, current, total, elapsed, estimatedRemaining)
		} else if progress != nil {
			progress(msg, calculatedPct)
		}
	}

	go func() {
		defer wg.Done()
		zhOnce.Do(func() {
			initZh(zhProgressEx)
			if pex != nil {
				pex("中文文本归一化引擎就绪", zhEnd, 0, 0, 0, 0)
			} else if progress != nil {
				progress("中文文本归一化引擎就绪", zhEnd)
			}
		})
	}()
	go func() {
		defer wg.Done()
		enOnce.Do(func() {
			initEn(enProgressEx)
			if pex != nil {
				pex("英文文本归一化引擎就绪", enEnd, 0, 0, 0, 0)
			} else if progress != nil {
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
	// 先检查缓存是否已存在，如果存在则跳过构建直接加载
	zhCacheOK := verifyCacheFiles(zhCacheDir, "zh_tn") == nil
	enCacheOK := verifyCacheFiles(enCacheDir, "en_tn") == nil

	if zhCacheOK && enCacheOK {
		log.Printf("[normalizer] FST 缓存文件已存在，跳过构建直接加载")
		// 用 rebuild=false 初始化，从磁盘加载缓存
		initZh(nil)
		initEn(nil)
		return nil
	}

	log.Printf("[normalizer] 开始构建 FST 缓存 (中文TN + 英文TN)")

	buildLang := func(name, cacheDir, prefix string, cacheExists bool, fn func(func(string, int, int))) error {
		if cacheExists {
			log.Printf("[normalizer] %s: 缓存已存在，跳过构建", name)
			return nil
		}
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

	if err := buildLang("中文TN", zhCacheDir, "zh_tn", zhCacheOK, func(p func(string, int, int)) {
		zhNormalizer = chinese.NewNormalizer(
			zhCacheDir, true,
			false, false, false, false, false, false,
			p,
		)
		zhReady = true
	}); err != nil {
		return fmt.Errorf("中文TN归一化缓存构建失败: %v", err)
	}

	if err := buildLang("英文TN", enCacheDir, "en_tn", enCacheOK, func(p func(string, int, int)) {
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

var EnglishVoices = map[string]bool{
	"Trump":  true,
	"Ava":    true,
	"Bella":  true,
	"Adam":   true,
	"Nathan": true,
}

func ResolveNormalizationLanguage(text string) string {
	return ResolveNormalizationLanguageWithVoice(text, "")
}

func ResolveNormalizationLanguageWithVoice(text string, voice string) string {
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
	if EnglishVoices[voice] {
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
	zhOnce.Do(func() { initZh(nil) })
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
		enOnce.Do(func() { initEn(nil) })
		result = enNormalizer.Normalize(text)
	default:
		zhOnce.Do(func() { initZh(nil) })
		result = zhNormalizer.Normalize(text)
	}
	return strings.TrimSpace(result)
}

func PrepareTTSText(text string, enableRobust bool, enableWeText bool) string {
	return PrepareTTSTextWithVoice(text, enableRobust, enableWeText, "")
}

func PrepareTTSTextWithVoice(text string, enableRobust bool, enableWeText bool, voice string) string {
	if text == "" {
		return text
	}
	current := text
	if enableRobust {
		current = NormalizeRobust(current)
	}
	if enableWeText {
		lang := ResolveNormalizationLanguageWithVoice(current, voice)
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

// Close 释放文本归一化引擎的资源，包括停止后台缓存清理 goroutine。
// 调用 Close 后不应再使用归一化功能。多次调用 Close 是安全的。
func Close() {
	if zhNormalizer != nil {
		zhNormalizer.Close()
		zhNormalizer = nil
		zhReady = false
	}
	if enNormalizer != nil {
		enNormalizer.Close()
		enNormalizer = nil
		enReady = false
	}
}
