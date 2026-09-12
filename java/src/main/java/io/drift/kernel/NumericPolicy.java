package io.drift.kernel;

import io.drift.fmt.Formats;
import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rounding;

/**
 * Every arithmetic decision a matmul makes, as data.
 *
 * <p>On a GPU these are not separable: the format the inputs are cast to, the width
 * the products are accumulated in, the order they are added in, and the way tensors
 * are scaled all arrive together as one kernel. When a run diverges the blame lands
 * on whichever of them is currently fashionable. Here each one is a field, and a run
 * can be repeated with exactly one of them changed.
 */
public final class NumericPolicy {

    /** Width the products are summed in. */
    public enum Acc {
        /** Double. The reference. */
        FP64,
        /** Real IEEE binary32 through a Java float, not a simulation of one. */
        FP32,
        FP16,
        BF16;

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
        /** First to last, one running total. */
        SEQUENTIAL,
        /** Last to first. Any gap from SEQUENTIAL is error no format can be blamed for. */
        REVERSED,
        /** Recursive halving. */
        PAIRWISE,
        /** Chunks summed in the accumulator width, chunk totals in FP32, as tiled kernels do. */
        BLOCKED
    }

    /** How a tensor is scaled into range before it is cast. */
    public enum Scaling {
        /** Cast as-is. */
        NONE,
        /** One arbitrary float factor for the whole tensor. */
        PER_TENSOR,
        /** OCP Microscaling: a shared power-of-two exponent per block, held in E8M0. */
        MX_BLOCK
    }

    public final String name;
    /** Format for weights and activations entering a matmul. Null means no cast. */
    public final MiniFloat inputFormat;
    /** Format for output gradients entering a matmul. Null means same as inputFormat. */
    public final MiniFloat gradFormat;
    public final Rounding inputRounding;
    /** Rounding for gradient casts. Null means same as inputRounding. */
    public final Rounding gradRounding;
    public final Scaling scaling;
    public final int scaleBlock;
    /**
     * For PER_TENSOR: 0 scales from the current tensor. N &gt; 0 scales from the largest
     * amax seen over the previous N steps, which is delayed scaling.
     */
    public final int amaxHistory;
    /**
     * For MX_BLOCK: added to the OCP shared exponent. 0 is the specification, which
     * maps a block maximum into the top binade and can clamp it; 1 spends a bit of
     * precision to guarantee it never clamps.
     */
    public final int mxHeadroom;
    public final Acc acc;
    public final Order order;
    public final int accBlock;
    public final Rounding accRounding;
    /** Master weights and optimizer state rounded to FP32 after every update. */
    public final boolean masterFp32;

    private NumericPolicy(Builder b) {
        this.name = b.name;
        this.inputFormat = b.inputFormat;
        this.gradFormat = b.gradFormat;
        this.inputRounding = b.inputRounding;
        this.gradRounding = b.gradRounding;
        this.scaling = b.scaling;
        this.scaleBlock = b.scaleBlock;
        this.amaxHistory = b.amaxHistory;
        this.mxHeadroom = b.mxHeadroom;
        this.acc = b.acc;
        this.order = b.order;
        this.accBlock = b.accBlock;
        this.accRounding = b.accRounding;
        this.masterFp32 = b.masterFp32;
    }

    /** Rounding actually used for gradient casts. */
    public Rounding gradCastRounding() {
        return gradRounding != null ? gradRounding : inputRounding;
    }

    /** Format actually used for gradients. */
    public MiniFloat gradCastFormat() {
        return gradFormat != null ? gradFormat : inputFormat;
    }

    /** FP64 throughout, nothing cast, nothing rounded. Everything else is measured against it. */
    public static NumericPolicy exact() {
        return builder("exact").acc(Acc.FP64).masterFp32(false).build();
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public Builder toBuilder(String newName) {
        Builder b = new Builder(newName);
        b.inputFormat = inputFormat;
        b.gradFormat = gradFormat;
        b.inputRounding = inputRounding;
        b.gradRounding = gradRounding;
        b.scaling = scaling;
        b.scaleBlock = scaleBlock;
        b.amaxHistory = amaxHistory;
        b.mxHeadroom = mxHeadroom;
        b.acc = acc;
        b.order = order;
        b.accBlock = accBlock;
        b.accRounding = accRounding;
        b.masterFp32 = masterFp32;
        return b;
    }

    public String describe() {
        return "name=" + name
                + " input=" + (inputFormat == null ? "none" : inputFormat.name)
                + " grad=" + (gradCastFormat() == null ? "none" : gradCastFormat().name)
                + " inputRounding=" + inputRounding
                + " gradRounding=" + gradCastRounding()
                + " scaling=" + scaling
                + " scaleBlock=" + scaleBlock
                + " amaxHistory=" + amaxHistory
                + " mxHeadroom=" + mxHeadroom
                + " acc=" + acc
                + " order=" + order
                + " accBlock=" + accBlock
                + " accRounding=" + accRounding
                + " masterFp32=" + masterFp32;
    }

    @Override
    public String toString() {
        return name;
    }

    public static final class Builder {
        private final String name;
        private MiniFloat inputFormat = null;
        private MiniFloat gradFormat = null;
        private Rounding inputRounding = Rounding.NEAREST_EVEN;
        private Rounding gradRounding = null;
        private Scaling scaling = Scaling.NONE;
        private int scaleBlock = 32;
        private int amaxHistory = 0;
        private int mxHeadroom = 0;
        private Acc acc = Acc.FP32;
        private Order order = Order.SEQUENTIAL;
        private int accBlock = 32;
        private Rounding accRounding = Rounding.NEAREST_EVEN;
        private boolean masterFp32 = true;

        private Builder(String name) {
            this.name = name;
        }

        public Builder input(MiniFloat f) { this.inputFormat = f; return this; }
        public Builder grad(MiniFloat f) { this.gradFormat = f; return this; }
        public Builder inputRounding(Rounding r) { this.inputRounding = r; return this; }
        public Builder gradRounding(Rounding r) { this.gradRounding = r; return this; }
        public Builder scaling(Scaling s) { this.scaling = s; return this; }
        public Builder scaleBlock(int n) { this.scaleBlock = n; return this; }
        public Builder amaxHistory(int n) { this.amaxHistory = n; return this; }
        public Builder mxHeadroom(int n) { this.mxHeadroom = n; return this; }
        public Builder acc(Acc a) { this.acc = a; return this; }
        public Builder order(Order o) { this.order = o; return this; }
        public Builder accBlock(int n) { this.accBlock = n; return this; }
        public Builder accRounding(Rounding r) { this.accRounding = r; return this; }
        public Builder masterFp32(boolean b) { this.masterFp32 = b; return this; }

        public NumericPolicy build() {
            return new NumericPolicy(this);
        }
    }
}
