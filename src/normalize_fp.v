module int18_to_bf16_lzd #(
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
    wire [17:0] normalized;

    assign sign = acc[17];
    assign mag = sign ? -acc : acc;

    lzd18 lzd_inst (.x(mag), .lz(lz));

    assign exp_unbiased = 9'(17) - 9'(lz) - 9'(FRAC_BITS);

    assign normalized = mag << lz;

    assign mant = normalized[16:10];
    
    // Compute biased exponent once (reduces logic depth)
    wire signed [8:0] exp_biased;
    assign exp_biased = exp_unbiased + BF16_BIAS;
    
    always @(*) begin
        // Flattened conditional logic for reduced mux depth
        if (mag == 18'd0) begin
            bf16 = {sign, 15'd0}; // Zero or Negative Zero
        end else if (exp_biased[8] || exp_biased < 0) begin // Check sign bit for negative
            bf16 = {sign, 15'd0}; // Underflow (flush to zero)
        end else if (exp_biased > 255) begin
            bf16 = {sign, 8'hFF, 7'd0}; // Overflow (infinity)
        end else begin
            // Normal case: assemble BF16
            bf16 = {sign, exp_biased[7:0], mant};
        end
    end
endmodule

module lzd18 (
    input  wire [17:0] x,
    output reg  [4:0] lz 
);
    // Efficient casez-based priority encoder
    // Synthesizes to O(log n) tree instead of O(n) mux chain
    always @(*) begin
        casez (x)
            18'b1?????????????????: lz = 5'd0;
            18'b01????????????????: lz = 5'd1;
            18'b001???????????????: lz = 5'd2;
            18'b0001??????????????: lz = 5'd3;
            18'b00001?????????????: lz = 5'd4;
            18'b000001????????????: lz = 5'd5;
            18'b0000001???????????: lz = 5'd6;
            18'b00000001??????????: lz = 5'd7;
            18'b000000001?????????: lz = 5'd8;
            18'b0000000001????????: lz = 5'd9;
            18'b00000000001???????: lz = 5'd10;
            18'b000000000001??????: lz = 5'd11;
            18'b0000000000001?????: lz = 5'd12;
            18'b00000000000001????: lz = 5'd13;
            18'b000000000000001???: lz = 5'd14;
            18'b0000000000000001??: lz = 5'd15;
            18'b00000000000000001?: lz = 5'd16;
            18'b000000000000000001: lz = 5'd17;
            default:                lz = 5'd18;
        endcase
    end
endmodule