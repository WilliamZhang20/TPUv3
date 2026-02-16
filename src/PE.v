module PE (
    input  wire        clk,
    input  wire        rst,
    input  wire        clear,
    input  wire [7:0]  a_in,
    input  wire [7:0]  b_in,
    output reg  [7:0]  a_out,
    output reg  [7:0]  b_out,
    output wire [15:0] c_out
);

    // ----------------------- FP8 E4M3 decode -----------------------
    wire sign_a = a_in[7];
    wire sign_b = b_in[7];
    wire [3:0] exp_a = a_in[6:3];
    wire [3:0] exp_b = b_in[6:3];
    wire denorm_a = ~|exp_a;
    wire denorm_b = ~|exp_b;

    // Combine mantissa selection into single operation
    wire [3:0] mant_a = {~denorm_a, a_in[2:0]};
    wire [3:0] mant_b = {~denorm_b, b_in[2:0]};

    // ----------------------- Multiply & align -----------------------
    wire prod_sign = sign_a ^ sign_b;
    wire [7:0] mant_prod = mant_a * mant_b;   // 4×4 = 8-bit product

    localparam FRAC_BITS = 8;
    
    // Optimized shift calculation: merged 3 adders into 1 expression
    // shift_right = 6 - FRAC_BITS - (exp_a + exp_b - 14) = 20 - FRAC_BITS - exp_a - exp_b
    // With FRAC_BITS=8: shift_right = 12 - exp_a - exp_b (where denorm uses exp=1)
    wire signed [6:0] shift_right = 7'sd12 - 
                                    (denorm_a ? 7'sd1 : {3'sd0, exp_a}) - 
                                    (denorm_b ? 7'sd1 : {3'sd0, exp_b});
    
    // Simplified shifter with clamping
    reg [17:0] aligned_prod;
    always @(*) begin
        if (shift_right[6])  // Negative = left shift
            aligned_prod = (shift_right < -7'sd10) ? 18'h3FFFF : ({10'd0, mant_prod} << (-shift_right[3:0]));
        else  // Positive = right shift
            aligned_prod = (shift_right > 7'sd17) ? 18'd0 : ({10'd0, mant_prod} >> shift_right[4:0]);
    end

    // ----------------------- Accumulator (2's complement) -----------------------
    reg signed [17:0] acc;

    wire signed [17:0] signed_prod =
        prod_sign ? -aligned_prod : aligned_prod;

    always @(posedge clk) begin
        if (rst) begin
            a_out <= 8'd0;
            b_out <= 8'd0;
            acc <= 18'sd0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            
            if (clear)
                acc <= signed_prod; 
            else
                acc <= acc + signed_prod;
        end
    end

    // ----------------------- INT18 → BF16 (combinational) -----------------------
    wire [15:0] bf16_c;
    int18_to_bf16 convert (
        .acc(acc), 
        .bf16(bf16_c)
    );

    assign c_out = bf16_c;


endmodule