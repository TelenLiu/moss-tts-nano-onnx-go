package sampler

import (
	"math/bits"
)

// PCG64 是与 Python numpy.random.default_rng(seed) 兼容的随机数生成器。
// 底层使用 PCG64 算法（128位 LCG + XSL-RR 输出函数），
// 与 numpy 的 SeedSequence + PCG64 完全一致，相同种子产生相同随机值序列。
type PCG64 struct {
	// state 和 inc 都是 128 位无符号整数，用 (hi, lo) 两个 uint64 表示
	stateHi, stateLo uint64
	incHi, incLo     uint64
}

// PCG64 乘法常量 (128位): 0x2360ED051FC65DA44385DF649FCCF645
const (
	pcgMultHi uint64 = 0x2360ED051FC65DA4
	pcgMultLo uint64 = 0x4385DF649FCCF645
)

// SeedSequence 常量
const (
	ssInitA    uint32 = 0x43b0d7e5
	ssMultA    uint32 = 0x931e8875
	ssInitB    uint32 = 0x8b51f9dd
	ssMultB    uint32 = 0x58f38ded
	ssMixMultL uint32 = 0xca01f9dd
	ssMixMultR uint32 = 0x4973f715
	ssXShift          = 16
	ssPoolSize        = 4
)

// NewPCG64 创建一个与 numpy.random.default_rng(seed) 兼容的 PCG64 随机数生成器。
func NewPCG64(seed int64) *PCG64 {
	p := &PCG64{}
	p.initSeedSequence(uint64(seed))
	return p
}

// initSeedSequence 使用 numpy SeedSequence 算法将种子转换为 PCG64 的初始状态
func (p *PCG64) initSeedSequence(seed uint64) {
	// === 1. 将种子转换为 uint32 数组（小端序，最低位在前）===
	var entropy []uint32
	if seed == 0 {
		entropy = []uint32{0}
	} else {
		s := seed
		for s > 0 {
			entropy = append(entropy, uint32(s&0xFFFFFFFF))
			s >>= 32
		}
	}

	// === 2. 熵混合 ===
	pool := [ssPoolSize]uint32{}

	// 步骤1: 用 hashmix 将熵填入 pool
	hashConst := ssInitA
	for i := 0; i < ssPoolSize; i++ {
		var val uint32
		if i < len(entropy) {
			val = entropy[i]
		}
		pool[i], hashConst = hashmix(val, hashConst)
	}

	// 步骤2: 全交叉混合，让每个 pool 元素影响其他所有元素
	for iSrc := 0; iSrc < ssPoolSize; iSrc++ {
		for iDst := 0; iDst < ssPoolSize; iDst++ {
			if iSrc != iDst {
				var mixed uint32
				mixed, hashConst = hashmix(pool[iSrc], hashConst)
				pool[iDst] = mix(pool[iDst], mixed)
			}
		}
	}

	// 步骤3: 处理剩余熵（熵数组比 pool 长时）
	for iSrc := ssPoolSize; iSrc < len(entropy); iSrc++ {
		for iDst := 0; iDst < ssPoolSize; iDst++ {
			var mixed uint32
			mixed, hashConst = hashmix(entropy[iSrc], hashConst)
			pool[iDst] = mix(pool[iDst], mixed)
		}
	}

	// === 3. generate_state(4, uint64) ===
	// 生成 8 个 uint32 值，然后两两组合为 4 个 uint64（小端序）
	state32 := [8]uint32{}
	hashConst = ssInitB
	poolIdx := 0
	for iDst := 0; iDst < 8; iDst++ {
		dataVal := pool[poolIdx%ssPoolSize]
		poolIdx++
		dataVal ^= hashConst
		hashConst *= ssMultB
		dataVal *= hashConst
		dataVal ^= (dataVal >> ssXShift)
		state32[iDst] = dataVal
	}

	// 小端序组合为 uint64: state32[2*i] 是低32位, state32[2*i+1] 是高32位
	val := [4]uint64{}
	for i := 0; i < 4; i++ {
		val[i] = uint64(state32[2*i]) | (uint64(state32[2*i+1]) << 32)
	}

	// === 4. PCG64 初始化 ===
	// initstate = (val[0] << 64) | val[1]  (高字在前)
	// initseq   = (val[2] << 64) | val[3]
	initstateHi, initstateLo := val[0], val[1]
	initseqHi, initseqLo := val[2], val[3]

	// inc = (initseq << 1) | 1  (128位左移1位，置最低位为1，保证 inc 是奇数)
	p.incHi, p.incLo = shiftLeft1_128(initseqHi, initseqLo)
	p.incLo |= 1

	// state = 0，然后 step，加 initstate，再 step
	p.stateHi, p.stateLo = 0, 0
	p.step() // state = 0 * mult + inc = inc
	p.stateHi, p.stateLo = add128mod(p.stateHi, p.stateLo, initstateHi, initstateLo)
	p.step()
}

// step 推进 PCG64 状态: state = state * multiplier + inc (mod 2^128)
func (p *PCG64) step() {
	// state * multiplier (mod 2^128)
	hi, lo := mul128mod(p.stateHi, p.stateLo, pcgMultHi, pcgMultLo)
	// + inc (mod 2^128)
	p.stateHi, p.stateLo = add128mod(hi, lo, p.incHi, p.incLo)
}

// nextUint64 生成下一个 64 位随机数
func (p *PCG64) nextUint64() uint64 {
	p.step()

	// XSL-RR 输出函数
	// xorred = state_hi ^ state_lo
	xorred := p.stateHi ^ p.stateLo
	// rot = state_hi >> 58 (取高6位作为旋转量)
	rot := p.stateHi >> 58
	// 右旋转
	return rotr64(xorred, rot)
}

// Float64 返回 [0, 1) 范围的 float64 随机数，与 numpy random() 一致
func (p *PCG64) Float64() float64 {
	r := p.nextUint64()
	// 取高53位，除以 2^53
	return float64(r>>11) * (1.0 / 9007199254740992.0)
}

// Float32 返回 [0, 1) 范围的 float32 随机数
// 与 numpy 的 float(rng.random()) 即 np.float32(rng.random()) 一致
func (p *PCG64) Float32() float32 {
	return float32(p.Float64())
}

// === 辅助函数 ===

// hashmix 是 SeedSequence 的熵混合函数
// 返回 (混合后的值, 更新后的 hashConst)
func hashmix(value, hashConst uint32) (uint32, uint32) {
	value ^= hashConst
	hashConst *= ssMultA
	value *= hashConst
	value ^= (value >> ssXShift)
	return value, hashConst
}

// mix 是 SeedSequence 的交叉混合函数
func mix(x, y uint32) uint32 {
	result := ssMixMultL*x - ssMixMultR*y
	result ^= (result >> ssXShift)
	return result
}

// mul128mod 计算 (aHi, aLo) * (bHi, bLo) mod 2^128，返回 (hi, lo)
func mul128mod(aHi, aLo, bHi, bLo uint64) (uint64, uint64) {
	// bits.Mul64 返回 (hi, lo) — 高64位在前，低64位在后
	hi, lo := bits.Mul64(aLo, bLo)
	// 交叉项只影响高64位（因为它们左移64位后落入高位）
	hi += aHi * bLo // 乘法溢出自动取 mod 2^64
	hi += aLo * bHi
	return hi, lo
}

// add128mod 计算 (aHi, aLo) + (bHi, bLo) mod 2^128，返回 (hi, lo)
func add128mod(aHi, aLo, bHi, bLo uint64) (uint64, uint64) {
	lo := aLo + bLo
	hi := aHi + bHi
	if lo < aLo { // 低64位溢出，进位
		hi++
	}
	return hi, lo
}

// shiftLeft1_128 将 128 位数左移 1 位，返回 (hi, lo)
func shiftLeft1_128(hi, lo uint64) (uint64, uint64) {
	hi = (hi << 1) | (lo >> 63)
	lo = lo << 1
	return hi, lo
}

// rotr64 将 64 位值右旋转 rot 位
func rotr64(value, rot uint64) uint64 {
	rot &= 63
	if rot == 0 {
		return value
	}
	return (value >> rot) | (value << (64 - rot))
}
