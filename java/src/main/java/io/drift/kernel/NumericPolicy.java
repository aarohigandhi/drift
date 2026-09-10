package io.drift.kernel;

import io.drift.fmt.Formats;
import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rounding;

/**
 * Every arithmetic decision a matmul makes, as data.
 *
 * <p>This is the point of the project. On a GPU these are not separable: the format
 * the inputs are cast to, the width the products are accumulated in, the order they
 * are added in, and the way tensors are scaled all arrive together as one kernel,
 * so when a run diverges the blame lands on whichever of them is currently
 * fashionable. Here each one is a field, and a run can be repeated with exactly one
 * of them changed.
 *
 * <p>Accumulation order is the field that has no equivalent anywhere else. Addition
 * in floating point is not associative, so the order the products are summed in
 * changes the answer, and no library exposes it because on real hardware it is a
 * property of the kernel rather than a setting.
 */
public record NumericPolicy(
        String name,
        MiniFloat inputFormat,
        Rounding inputRounding,
        Scaling scaling,
        int scaleBlock,
        Acc acc,
        Order order,
        int accBlock,
        Rounding accRounding
) {

    /** Width the products are summed in. */
    public enum Acc {
        /** Double. The reference: everything else is measured as drift away from it. */
        FP64,
        /**
         * Real IEEE binary32, via a Java float, not a simulation of one. What a
         * tensor core accumulates in.
         */
        FP32,
        FP16,
        BF16;

        /** The format to round through, or null when the machine already does it. */
        public MiniFloat format() {
            return switch (this) {
                case FP64, FP32 -> null;
                case FP16 -> Formats.FP16;
                case BF16 -> Formats.BF16;
            };
        }
    }

    /** The order the products are summed in. */
    public enum Order {
        /** First to last, one running total. What a naive loop does. */
        SEQUENTIAL,
        /**
         * Last to first. Numerically no more or less defensible than SEQUENTIAL,
         * and included precisely for that reason: any gap between the two is
         * error that no choice of format can be blamed for.
         */
        REVERSED,
        /** Recursive halving. Error grows with log n rather than n. */
        PAIRWISE,
        /**
         * Sum in chunks, each chunk in the accumulator width, chunk totals in FP32.
         * What hardware actually does when it splits the K dimension across tiles,
         * and the reason a matmul can change answer when only the tile size changes.
         */
        BLOCKED
    }

    /** How a tensor is scaled into range before it is cast. */
    public enum Scaling {
        /** Cast as-is. Anything outside the format saturates. */
        NONE,
        /**
         * One factor for the whole tensor, chosen so its largest element lands on
         * the largest representable value. The factor is an arbitrary float, so
         * multiplying by it is itself a rounding.
         */
        PER_TENSOR,
        /**
         * OCP Microscaling: one shared exponent per block of elements, held in E8M0.
         * The factor is a power of two, so unlike PER_TENSOR the scaling itself is
         * exact and the only error is the cast.
         */
        MX_BLOCK
    }

    /** FP64 throughout, no scaling, no quantisation. The thing everything is compared to. */
    public static NumericPolicy exact() {
        return new NumericPolicy("exact", null, Rounding.NEAREST_EVEN,
                Scaling.NONE, 32, Acc.FP64, Order.SEQUENTIAL, 128, Rounding.NEAREST_EVEN);
    }

    /** BF16 inputs into an FP32 accumulator. The ordinary way to train today. */
    public static NumericPolicy bf16Baseline() {
        return new NumericPolicy("bf16", Formats.BF16, Rounding.NEAREST_EVEN,
                Scaling.NONE, 32, Acc.FP32, Order.SEQUENTIAL, 128, Rounding.NEAREST_EVEN);
    }

    /** E4M3 inputs, per-tensor scaled, into an FP32 accumulator. The usual FP8 recipe. */
    public static NumericPolicy fp8Baseline() {
        return new NumericPolicy("fp8-e4m3", Formats.E4M3, Rounding.NEAREST_EVEN,
                Scaling.PER_TENSOR, 32, Acc.FP32, Order.SEQUENTIAL, 128, Rounding.NEAREST_EVEN);
    }

    /** MXFP4: E2M1 elements under a shared power-of-two exponent every 32 values. */
    public static NumericPolicy mxfp4() {
        return new NumericPolicy("mxfp4", Formats.E2M1, Rounding.NEAREST_EVEN,
                Scaling.MX_BLOCK, 32, Acc.FP32, Order.SEQUENTIAL, 128, Rounding.NEAREST_EVEN);
    }

    /** True when no quantisation happens at all and the accumulator is FP64. */
    public boolean isExact() {
        return inputFormat == null && acc == Acc.FP64;
    }

    /** A copy with one field changed, which is how the sweep is built. */
    public NumericPolicy with(String newName, Acc newAcc, Order newOrder) {
        return new NumericPolicy(newName, inputFormat, inputRounding, scaling, scaleBlock,
                newAcc, newOrder, accBlock, accRounding);
    }

    public NumericPolicy withOrder(String newName, Order newOrder, int newAccBlock) {
        return new NumericPolicy(newName, inputFormat, inputRounding, scaling, scaleBlock,
                acc, newOrder, newAccBlock, accRounding);
    }

    public NumericPolicy withRounding(String newName, Rounding newInput, Rounding newAcc) {
        return new NumericPolicy(newName, inputFormat, newInput, scaling, scaleBlock,
                acc, order, accBlock, newAcc);
    }

    public NumericPolicy withScaling(String newName, Scaling newScaling, int newBlock) {
        return new NumericPolicy(newName, inputFormat, inputRounding, newScaling, newBlock,
                acc, order, accBlock, accRounding);
    }

    @Override
    public String toString() {
        return name;
    }
}
