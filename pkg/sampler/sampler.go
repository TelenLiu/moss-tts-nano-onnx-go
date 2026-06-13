package sampler

import (
	"math"
	"math/rand"
	"strings"
)

const (
	SampleModeGreedy = "greedy"
	SampleModeFixed  = "fixed"
	SampleModeFull   = "full"
	SampleModeHybrid = "hybrid" // 混合模式：前N帧用full保证音色克隆质量，后续帧用fixed加速
)

func Argmax(values []float32) int {
	bestIndex := 0
	bestValue := float32(math.Inf(-1))
	for i, v := range values {
		if v > bestValue {
			bestValue = v
			bestIndex = i
		}
	}
	return bestIndex
}

func ArgmaxWithRepetitionPenalty(values []float32, previousTokenSet map[int]bool, penalty float32) int {
	bestIndex := 0
	bestValue := float64(math.Inf(-1))
	applyPenalty := len(previousTokenSet) > 0 && penalty != 1.0
	for i, v := range values {
		score := float64(v)
		if applyPenalty && previousTokenSet[i] {
			if score < 0 {
				score *= float64(penalty)
			} else {
				score /= float64(penalty)
			}
		}
		if score > bestValue {
			bestValue = score
			bestIndex = i
		}
	}
	return bestIndex
}

func ApplyRepetitionPenalty(values []float32, previousTokenIDs []int, penalty float32) []float32 {
	if len(previousTokenIDs) == 0 || penalty == 1.0 {
		return values
	}
	result := make([]float32, len(values))
	copy(result, values)
	seen := make(map[int]bool)
	for _, id := range previousTokenIDs {
		if id < 0 || id >= len(result) || seen[id] {
			continue
		}
		seen[id] = true
		if result[id] > 0 {
			result[id] /= penalty
		} else {
			result[id] *= penalty
		}
	}
	return result
}

func Softmax(values []float64) []float64 {
	if len(values) == 0 {
		return nil
	}
	maxVal := values[0]
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}
	exps := make([]float64, len(values))
	var sum float64
	for i, v := range values {
		exps[i] = math.Exp(v - maxVal)
		sum += exps[i]
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

type indexedVal struct {
	idx int
	val float64
}

func SampleFromScores(values []float32, doSample bool, temperature float64, topK int, topP float64, rng *rand.Rand) int {
	if !doSample {
		return Argmax(values)
	}
	if temperature <= 0 {
		temperature = 1.0
	}
	scores := make([]float64, len(values))
	for i, v := range values {
		scores[i] = float64(v) / temperature
	}
	if topK > 0 && topK < len(scores) {
		threshold := nthLargest(scores, topK)
		for i, s := range scores {
			if s < threshold {
				scores[i] = math.Inf(-1)
			}
		}
	}
	if topP > 0 && topP < 1 {
		indexed := make([]indexedVal, len(scores))
		for i, s := range scores {
			indexed[i] = indexedVal{i, s}
		}
		sortDescIndexed(indexed)
		sortedScores := make([]float64, len(indexed))
		for i, item := range indexed {
			sortedScores[i] = item.val
		}
		probs := Softmax(sortedScores)
		removeMask := make([]bool, len(indexed))
		var cumulative float64
		for i, p := range probs {
			cumulative += p
			if cumulative > topP {
				removeMask[i] = true
			}
		}
		// 与 Python 实现一致：右移 removeMask 一位，确保概率最高的 token 永远不被移除
		// Python: for index in range(len(remove_mask) - 1, 0, -1): remove_mask[index] = remove_mask[index - 1]
		// Python: remove_mask[0] = False
		for i := len(removeMask) - 1; i > 0; i-- {
			removeMask[i] = removeMask[i-1]
		}
		removeMask[0] = false
		for i, mask := range removeMask {
			if mask {
				scores[indexed[i].idx] = math.Inf(-1)
			}
		}
	}
	probabilities := Softmax(scores)
	randomValue := rng.Float64()
	for i, p := range probabilities {
		randomValue -= p
		if randomValue <= 0 {
			return i
		}
	}
	return Argmax(values)
}

func SampleAssistantTextToken(textLogits []float32, assistantSlotTokenID, audioEndTokenID int, doSample bool, temperature float64, topK int, topP float64, rng *rand.Rand) int {
	candidateIDs := []int{assistantSlotTokenID, audioEndTokenID}
	candidateScores := []float32{textLogits[assistantSlotTokenID], textLogits[audioEndTokenID]}
	effectiveTopK := topK
	if effectiveTopK > len(candidateScores) {
		effectiveTopK = len(candidateScores)
	}
	sampledIndex := SampleFromScores(candidateScores, doSample, temperature, effectiveTopK, topP, rng)
	return candidateIDs[sampledIndex]
}

func SampleAudioToken(audioLogits []float32, previousTokenIDs []int, previousTokenSet map[int]bool, doSample bool, temperature float64, topK int, topP float64, repetitionPenalty float32, rng *rand.Rand) int {
	if !doSample {
		return ArgmaxWithRepetitionPenalty(audioLogits, previousTokenSet, repetitionPenalty)
	}
	penalizedScores := ApplyRepetitionPenalty(audioLogits, previousTokenIDs, repetitionPenalty)
	return SampleFromScores(penalizedScores, true, temperature, topK, topP, rng)
}

func NormalizeSampleMode(rawSampleMode string, rawDoSample bool) string {
	normalized := strings.TrimSpace(rawSampleMode)
	switch normalized {
	case SampleModeGreedy, SampleModeFixed, SampleModeFull:
		return normalized
	case "mixed3":
		if rawDoSample {
			return SampleModeFixed
		}
		return SampleModeGreedy
	default:
		if !rawDoSample {
			return SampleModeGreedy
		}
		return SampleModeFixed
	}
}

func nthLargest(scores []float64, n int) float64 {
	if n <= 0 || n > len(scores) {
		return math.Inf(-1)
	}
	sorted := make([]float64, len(scores))
	copy(sorted, scores)
	for i := 0; i < n && i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] > sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	if n-1 < len(sorted) {
		return sorted[n-1]
	}
	return math.Inf(-1)
}

func sortDescIndexed(items []indexedVal) {
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j].val > items[j-1].val; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}
}
