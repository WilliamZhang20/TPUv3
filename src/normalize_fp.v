module int18_to_bf16 #(
    parameter FRAC_BITS = 8
)(
    input  wire signed [17:0] acc,
    output reg  [15:0] bf16
);
    localparam signed [8:0] BF16_BIAS = 9'sd127;  // Signed to preserve signed arithmetic
    localparam SHIFT_BASE = 17 - FRAC_BITS;  // Precompute: 17 - 8 = 9
    
    wire sign;
    wire [17:0] mag;
    wire [4:0] lz;
    wire signed [8:0] exp_unbiased;
    reg [7:0] exp;
    wire [6:0] mant;
    // verilator lint_off UNUSEDSIGNAL
    wire [17:0] normalized;
    // verilator lint_on UNUSEDSIGNAL

    assign sign = acc[17];
    assign mag = sign ? -acc : acc;

    lzd18 lzd_inst (.x(mag), .lz(lz));

    assign exp_unbiased = 9'(SHIFT_BASE) - 9'(lz);  // Simplified: one subtraction instead of two

    // Normalize mantissa - only bits [16:10] used for BF16 7-bit mantissa
    // Bit [17] is implicit leading 1, bits [9:0] are discarded for precision
    assign normalized = mag << lz;

    assign mant = normalized[16:10];
    
    always @(*) begin
        exp = 8'b0;
        if (mag == 18'd0) begin
            bf16 = {sign, 15'd0}; // Zero or Negative Zero
        end else begin
            if (exp_unbiased + BF16_BIAS < 0) begin
                bf16 = {sign, 15'd0}; // Underflow (flush to zero)
            end else if (exp_unbiased + BF16_BIAS > 255) begin
                bf16 = {sign, 8'hFF, 7'd0}; // Overflow (infinity)
            end else begin
                // Convert to biased exponent and assemble BF16
                exp = 8'(exp_unbiased + BF16_BIAS);  // Explicit cast - already range-checked
                bf16 = {sign, exp, mant};
            end
        end
    end
endmodule

module lzd18 (
    input  wire [17:0] x,
    output reg  [4:0] lz 
);

    // Balanced binary tree LZD for 18 bits
    // Level 1: 6 groups of 3 bits each
    wire [2:0] lz_g0, lz_g1, lz_g2, lz_g3, lz_g4, lz_g5;
    wire any_g0, any_g1, any_g2, any_g3, any_g4, any_g5;
    
    // Group 0: bits [17:15]
    assign any_g0 = |x[17:15];
    assign lz_g0 = x[17] ? 3'd0 : (x[16] ? 3'd1 : (x[15] ? 3'd2 : 3'd3));
    
    // Group 1: bits [14:12]
    assign any_g1 = |x[14:12];
    assign lz_g1 = x[14] ? 3'd0 : (x[13] ? 3'd1 : (x[12] ? 3'd2 : 3'd3));
    
    // Group 2: bits [11:9]
    assign any_g2 = |x[11:9];
    assign lz_g2 = x[11] ? 3'd0 : (x[10] ? 3'd1 : (x[9] ? 3'd2 : 3'd3));
    
    // Group 3: bits [8:6]
    assign any_g3 = |x[8:6];
    assign lz_g3 = x[8] ? 3'd0 : (x[7] ? 3'd1 : (x[6] ? 3'd2 : 3'd3));
    
    // Group 4: bits [5:3]
    assign any_g4 = |x[5:3];
    assign lz_g4 = x[5] ? 3'd0 : (x[4] ? 3'd1 : (x[3] ? 3'd2 : 3'd3));
    
    // Group 5: bits [2:0]
    assign any_g5 = |x[2:0];
    assign lz_g5 = x[2] ? 3'd0 : (x[1] ? 3'd1 : (x[0] ? 3'd2 : 3'd3));
    
    // Level 2: Balanced binary tree selection (3 pairs)
    wire any_p0, any_p1, any_p2;  // Pair presence
    wire [4:0] lz_p0, lz_p1, lz_p2;  // Pair results with offset
    
    // Pair 0: groups 0-1
    assign any_p0 = any_g0 | any_g1;
    assign lz_p0 = any_g0 ? {2'd0, lz_g0} : (5'd3 + {2'd0, lz_g1});
    
    // Pair 1: groups 2-3
    assign any_p1 = any_g2 | any_g3;
    assign lz_p1 = any_g2 ? (5'd6 + {2'd0, lz_g2}) : (5'd9 + {2'd0, lz_g3});
    
    // Pair 2: groups 4-5
    assign any_p2 = any_g4 | any_g5;
    assign lz_p2 = any_g4 ? (5'd12 + {2'd0, lz_g4}) : (5'd15 + {2'd0, lz_g5});
    
    // Level 3: Final binary selection (3-way requires 2 stages)
    wire any_half0;  // pairs 0-1
    wire [4:0] lz_half0;
    
    assign any_half0 = any_p0 | any_p1;
    assign lz_half0 = any_p0 ? lz_p0 : lz_p1;
    
    // Final output
    always @(*) begin
        if (any_half0)
            lz = lz_half0;
        else if (any_p2)
            lz = lz_p2;
        else
            lz = 5'd18;  // All zeros
    end
    
endmodule