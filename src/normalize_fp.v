module int18_to_bf16 #(
    parameter FRAC_BITS = 8
)(
    input  wire signed [17:0] acc,
    output reg  [15:0] bf16
);
    localparam BF16_BIAS = 127;
    
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

    assign exp_unbiased = 9'(17) - 9'(lz) - 9'(FRAC_BITS);

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
                exp = exp_unbiased + BF16_BIAS;
                bf16 = {sign, exp, mant};
            end
        end
    end
endmodule

module lzd18 (
    input  wire [17:0] x,
    output reg  [4:0] lz 
);

    // Tree-based leading zero detector for reduced critical path
    // Split into 6 groups of 3 bits each (bits 17:15, 14:12, 11:9, 8:6, 5:3, 2:0)
    
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
    
    // Second level: select which group and add offset
    always @(*) begin
        if (any_g0)
            lz = {2'd0, lz_g0};
        else if (any_g1)
            lz = 5'd3 + {2'd0, lz_g1};
        else if (any_g2)
            lz = 5'd6 + {2'd0, lz_g2};
        else if (any_g3)
            lz = 5'd9 + {2'd0, lz_g3};
        else if (any_g4)
            lz = 5'd12 + {2'd0, lz_g4};
        else if (any_g5)
            lz = 5'd15 + {2'd0, lz_g5};
        else
            lz = 5'd18; // All zeros
    end
    
endmodule