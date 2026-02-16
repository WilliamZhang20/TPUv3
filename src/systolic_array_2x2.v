module systolic_array_2x2 (
    input wire clk,
    input wire rst,
    input wire clear,
    input wire activation,

    input wire [7:0] weight0, weight1, weight2, weight3,
    input wire [7:0] input0, input1, input2, input3,

    input wire [1:0] a0_sel, a1_sel, b0_sel, b1_sel,
    input wire transpose,

    output wire signed [15:0] c00, c01, c10, c11
);

    wire [7:0] a_wire [0:1][0:2];
    wire [7:0] b_wire [0:2][0:1];
    wire signed [15:0] c_array [0:1][0:1];

    assign a_wire[0][0] = a0_sel[1] ? 8'b0 : (a0_sel[0] ? weight1 : weight0);
    assign a_wire[1][0] = a1_sel[1] ? 8'b0 : (a1_sel[0] ? weight3 : weight2);

    wire [7:0] in_cross0 = transpose ? input1 : input2;
    wire [7:0] in_cross1 = transpose ? input2 : input1;

    assign b_wire[0][0] = b0_sel[1] ? 8'b0 : (b0_sel[0] ? in_cross0 : input0);
    assign b_wire[0][1] = b1_sel[1] ? 8'b0 : (b1_sel[0] ? input3   : in_cross1);

    genvar i, j;
    generate
        for (i = 0; i < 2; i = i + 1) begin : row
            for (j = 0; j < 2; j = j + 1) begin : col
                PE pe_inst (
                    .clk(clk),
                    .rst(rst),
                    .clear(clear),
                    .a_in(a_wire[i][j]),
                    .b_in(b_wire[i][j]),
                    .a_out(a_wire[i][j+1]),
                    .b_out(b_wire[i+1][j]),
                    .c_out(c_array[i][j])
                );
            end
        end
    endgenerate

    wire signed [15:0] zero = 16'sd0;

    assign c00 = (activation && c_array[0][0] < 0) ? zero : c_array[0][0];
    assign c01 = (activation && c_array[0][1] < 0) ? zero : c_array[0][1];
    assign c10 = (activation && c_array[1][0] < 0) ? zero : c_array[1][0];
    assign c11 = (activation && c_array[1][1] < 0) ? zero : c_array[1][1];

endmodule