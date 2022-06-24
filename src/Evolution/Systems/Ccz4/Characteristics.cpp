// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/Characteristics.hpp"

#include <algorithm>  // IWYU pragma: keep
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Ccz4 {

template <size_t Dim, typename Frame>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::II<DataVector, Dim, Frame>& inverse_conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const double cleaning_speed) {
  // \lambda_1 to \lambda_{21}
  (*char_speeds)[0] = 0.0;
  // \lambda_{24} to \lambda_{29}
  (*char_speeds)[2] = sqrt(get<0, 0>(inverse_conformal_spatial_metric)) *
                      get(conformal_factor) * get(lapse);
  // \lambda_{30} to \lambda_{25}
  (*char_speeds)[3] = -(*char_speeds)[2];
  // \lambda_{22}, \lambda_{23} TODO : + or - ?
  (*char_speeds)[1] = e * (*char_speeds)[2];
}

template <size_t Dim, typename Frame>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::II<DataVector, Dim, Frame>& inverse_conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const double cleaning_speed) {
  auto char_speeds =
      make_with_value<typename Tags::CharacteristicSpeeds<Dim, Frame>::type>(
          get(lapse), 0.);
  characteristic_speeds(make_not_null(&char_speeds),
                        inverse_conformal_spatial_metric, conformal_factor,
                        lapse, cleaning_speed);
  return char_speeds;
}

template <size_t Dim, typename Frame>
void characteristic_fields(
    const gsl::not_null<typename Tags::CharacteristicFields<Dim, Frame>::type*>
        char_fields,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame>& inverse_conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const double cleaning_speed) {
  const auto& lo_11 = get<0, 0>(conformal_spatial_metric);
  const auto& lo_12 = get<0, 1>(conformal_spatial_metric);
  const auto& lo_13 = get<0, 2>(conformal_spatial_metric);
  const auto& lo_22 = get<1, 1>(conformal_spatial_metric);
  const auto& lo_23 = get<1, 2>(conformal_spatial_metric);
  const auto& lo_33 = get<2, 2>(conformal_spatial_metric);

  const auto& up_11 = get<0, 0>(inverse_conformal_spatial_metric);
  const auto& up_12 = get<0, 1>(inverse_conformal_spatial_metric);
  const auto& up_13 = get<0, 2>(inverse_conformal_spatial_metric);
  const auto& up_22 = get<1, 1>(inverse_conformal_spatial_metric);
  const auto& up_23 = get<1, 2>(inverse_conformal_spatial_metric);
  const auto& up_33 = get<2, 2>(inverse_conformal_spatial_metric);

  auto& r1 = (*char_speeds)[0];
  auto& r2 = (*char_speeds)[1];
  auto& r3 = (*char_speeds)[2];
  auto& r4 = (*char_speeds)[3];
  auto& r5 = (*char_speeds)[4];
  auto& r6 = (*char_speeds)[5];
  auto& r7 = (*char_speeds)[6];
  auto& r8 = (*char_speeds)[7];
  auto& r9 = (*char_speeds)[8];
  auto& r10 = (*char_speeds)[9];
  auto& r11 = (*char_speeds)[10];
  auto& r12 = (*char_speeds)[11];
  auto& r13 = (*char_speeds)[12];
  auto& r14 = (*char_speeds)[13];
  auto& r15 = (*char_speeds)[14];
  auto& r16 = (*char_speeds)[15];
  auto& r17 = (*char_speeds)[16];
  auto& r18 = (*char_speeds)[17];
  auto& r19 = (*char_speeds)[18];
  auto& r20 = (*char_speeds)[19];
  auto& r21 = (*char_speeds)[20];
  auto& r22 = (*char_speeds)[21];
  auto& r23 = (*char_speeds)[22];
  auto& r24 = (*char_speeds)[23];
  auto& r25 = (*char_speeds)[24];
  auto& r26 = (*char_speeds)[25];
  auto& r27 = (*char_speeds)[26];
  auto& r28 = (*char_speeds)[27];
  auto& r29 = (*char_speeds)[28];
  auto& r30 = (*char_speeds)[29];
  auto& r31 = (*char_speeds)[30];
  auto& r32 = (*char_speeds)[31];
  auto& r33 = (*char_speeds)[32];
  auto& r34 = (*char_speeds)[33];
  auto& r35 = (*char_speeds)[34];

  // 1
  r1[0] = lo_11 / lo_33;
  r1[1] = lo_11 / lo_33;
  r1[2] = lo_11 / lo_33;
  r1[3] = lo_11 / lo_33;
  r1[4] = lo_11 / lo_33;
  r1[5] = 1.0;

  for (size_t i = 6; i < char_fields.size(); i++) {
    r1[i] = 0.0;
  }

  // 2
  for (size_t i = 0; i < 8; i++) {
    r2[i] = 0.0;
  }

  r2[8] = square(up_11);
  r2[9] = up_11 * up_12;
  r2[10] = up_11 * up_13;

  r2[11] = 0.0;
  r2[12] = 0.0;
  r2[13] = 0.0;

  r2[14] = 1.0;

  for (size_t i = 15; i < char_fields.size(); i++) {
    r2[i] = 0.0;
  }

  // 3
  for (size_t i = 0; i < 8; i++) {
    r3[i] = 0.0;
  }

  r3[8] = 2.0 * r2[9];
  r3[9] = 2.0 * up_11 * up_22;
  r3[10] = 2.0 * up_11 * up_23;

  r3[11] = 0.0;
  r3[12] = 0.0;
  r3[13] = 0.0;
  r3[14] = 0.0;

  r3[15] = 1.0;

  for (size_t i = 16; i < char_fields.size(); i++) {
    r3[i] = 0.0;
  }

  // 4
  for (size_t i = 0; i < 8; i++) {
    r4[i] = 0.0;
  }

  r4[8] = 2.0 * r2[10];
  r4[9] = r3[10];
  r4[10] = 2.0 * up_11 * up_33;

  r4[11] = 0.0;
  r4[12] = 0.0;
  r4[13] = 0.0;
  r4[14] = 0.0;
  r4[15] = 0.0;

  r4[16] = 1.0;

  for (size_t i = 17; i < char_fields.size(); i++) {
    r4[i] = 0.0;
  }

  // 5
  for (size_t i = 0; i < 9; i++) {
    r5[i] = 0.0;
  }

  r5[9] = lo_33 / up_11;
  r5[10] = -lo_23 / up_11;
  r5[11] = -up_12 / up_11;

  r5[12] = 1.0;

  for (size_t i = 13; i < char_fields.size(); i++) {
    r5[i] = 0.0;
  }

  // 6
  for (size_t i = 0; i < 9; i++) {
    r6[i] = 0.0;
  }

  r6[9] = r5[10];
  r6[10] = lo_22 / up_11;
  r6[11] = -up_13 / up_11;

  r6[12] = 0.0;

  r6[13] = 1.0;

  for (size_t i = 14; i < char_fields.size(); i++) {
    r6[i] = 0.0;
  }

  // 7
  for (size_t i = 0; i < 8; i++) {
    r7[i] = 0.0;
  }

  r7[8] = r2[9];
  r7[9] = up_12 * up_12;
  r7[10] = up_12 * up_13;

  for (size_t i = 11; i < 20; i++) {
    r7[i] = 0.0;
  }

  r7[20] = 1.0;

  for (size_t i = 21; i < char_fields.size(); i++) {
    r7[i] = 0.0;
  }

  // 8
  for (size_t i = 0; i < 8; i++) {
    r8[i] = 0.0;
  }

  r8[8] = 2.0 * r7[9];
  r8[9] = 2.0 * up_12 * up_22;
  r8[10] = 2.0 * up_12 * up_23;

  for (size_t i = 11; i < 21; i++) {
    r8[i] = 0.0;
  }

  r8[21] = 1.0;

  for (size_t i = 22; i < char_fields.size(); i++) {
    r8[i] = 0.0;
  }

  // 9
  for (size_t i = 0; i < 8; i++) {
    r9[i] = 0.0;
  }

  r9[8] = 2.0 * r7[10];
  r9[9] = r8[10];
  r9[10] = 2.0 * up_12 * up_33;

  for (size_t i = 11; i < 22; i++) {
    r9[i] = 0.0;
  }

  r9[22] = 1.0;

  for (size_t i = 23; i < char_fields.size(); i++) {
    r9[i] = 0.0;
  }

  // 10
  for (size_t i = 0; i < 17; i++) {
    r10[i] = 0.0;
  }

  r10[17] = r5[11];

  for (size_t i = 18; i < 23; i++) {
    r10[i] = 0.0;
  }

  r10[23] = 1.0;

  for (size_t i = 24; i < char_fields.size(); i++) {
    r10[i] = 0.0;
  }

  // 11
  for (size_t i = 0; i < 18; i++) {
    r11[i] = 0.0;
  }

  r11[18] = r5[11];

  for (size_t i = 19; i < 24; i++) {
    r11[i] = 0.0;
  }

  r11[24] = 1.0;

  for (size_t i = 25; i < char_fields.size(); i++) {
    r11[i] = 0.0;
  }

  // 12
  for (size_t i = 0; i < 19; i++) {
    r12[i] = 0.0;
  }

  r12[19] = r5[11];

  for (size_t i = 20; i < 25; i++) {
    r12[i] = 0.0;
  }

  r12[25] = 1.0;

  for (size_t i = 26; i < char_fields.size(); i++) {
    r12[i] = 0.0;
  }

  // 13
  for (size_t i = 0; i < 8; i++) {
    r13[i] = 0.0;
  }

  r13[8] = r2[10];
  r13[9] = r7[10];
  r13[10] = square(up_13);

  for (size_t i = 11; i < 26; i++) {
    r13[i] = 0.0;
  }

  r13[26] = 1.0;

  for (size_t i = 27; i < char_fields.size(); i++) {
    r13[i] = 0.0;
  }

  // 14
  for (size_t i = 0; i < 8; i++) {
    r14[i] = 0.0;
  }

  r14[8] = r13[9];
  r14[9] = 2.0 * up_13 * up_22;
  r14[10] = 2.0 * up_13 * up_23;

  for (size_t i = 11; i < 27; i++) {
    r14[i] = 0.0;
  }

  r14[27] = 1.0;

  for (size_t i = 28; i < char_fields.size(); i++) {
    r14[i] = 0.0;
  }

  // 15
  for (size_t i = 0; i < 8; i++) {
    r15[i] = 0.0;
  }

  r15[8] = 2.0 * r13[10];
  r15[9] = r14[10];
  r15[10] = 2.0 * up_13 * up_33;

  for (size_t i = 11; i < 28; i++) {
    r15[i] = 0.0;
  }

  r15[28] = 1.0;

  for (size_t i = 29; i < char_fields.size(); i++) {
    r15[i] = 0.0;
  }

  // 16
  for (size_t i = 0; i < 17; i++) {
    r16[i] = 0.0;
  }

  r16[17] = r6[11];

  for (size_t i = 18; i < 29; i++) {
    r16[i] = 0.0;
  }

  r16[29] = 1.0;

  for (size_t i = 30; i < char_fields.size(); i++) {
    r16[i] = 0.0;
  }

  // 17
  for (size_t i = 0; i < 18; i++) {
    r17[i] = 0.0;
  }

  r17[18] = r6[11];

  for (size_t i = 19; i < 30; i++) {
    r17[i] = 0.0;
  }

  r17[30] = 1.0;

  for (size_t i = 31; i < char_fields.size(); i++) {
    r17[i] = 0.0;
  }

  // 18
  for (size_t i = 0; i < 19; i++) {
    r18[i] = 0.0;
  }

  r18[19] = r6[11];

  for (size_t i = 20; i < 31; i++) {
    r18[i] = 0.0;
  }

  r18[31] = 1.0;

  for (size_t i = 32; i < char_fields.size(); i++) {
    r18[i] = 0.0;
  }

  // 19 -21 setup
  for (size_t i = 0; i < 8; i++) {
    r19[i] = 0.0;
    r20[i] = 0.0;
    r21[i] = 0.0;
  }

  r19[8] = 2.0 * lo_11 * up_11 + 3.0 * lo_12 * up_12 + 3.0 * lo_13 * up_13;
  r19[9] = lo_12 * up_12 + lo_22 * up_22 + lo_23 * up_23;
  r19[10] = lo_13 * up_13 + lo_23 * up_23 + lo_33 * up_33;

  r20[8] = r19[8];
  r20[9] = r19[9];
  r20[10] = r19[10];

  r21[8] = r19[8];
  r21[9] = r19[9];
  r21[10] = r19[10];

  // 19 cont'd
  r19[8] *= -up_11;
  r19[9] = -up_12 * r19[9] + lo_11 * up_12 * up_11;
  r19[10] = -up_13 * r19[10] + lo_11 * up_13 * up_11;

  for (size_t i = 11; i < 17; i++) {
    r19[i] = 0.0;
  }

  r19[17] = lo_22;
  r19[18] = lo_23;
  r19[19] = lo_33;

  for (size_t i = 20; i < 32; i++) {
    r19[i] = 0.0;
  }

  r19[32] = 1.0;

  r19[33] = 0.0;
  r19[34] = 0.0;

  // 20 cont'd
  r20[8] *= -up_12;
  r20[9] = -up_22 * r20[9] + lo_11 * square(up_12);
  r20[10] = -up_23 * r20[10] + lo_11 * up_13 * up_12;

  for (size_t i = 11; i < 17; i++) {
    r20[i] = 0.0;
  }

  r20[17] = -r5[11];
  r20[18] = r20[17];
  r20[19] = r20[17];
  r20[17] *= lo_22;
  r20[18] *= lo_23;
  r20[19] *= lo_23;

  for (size_t i = 20; i < 33; i++) {
    r20[i] = 0.0;
  }

  r20[33] = 1.0;

  r20[34] = 0.0;

  // 21 cont'd
  r21[8] *= -up_13;
  r21[9] = -up_23 * r21[9] + lo_11 * up_12 * up_13;
  r21[10] = -up_33 * r21[10] + lo_11 * square(up_13);

  for (size_t i = 11; i < 17; i++) {
    r21[i] = 0.0;
  }

  r21[17] = -r6[11];
  r21[18] = r21[17];
  r21[19] = r21[17];
  r21[17] *= lo_22;
  r21[18] *= lo_23;
  r21[19] *= lo_23;

  for (size_t i = 20; i < 34; i++) {
    r21[i] = 0.0;
  }

  r21[34] = 1.0;

  // 22
  for (size_t i = 0; i < 7; i++) {
    r22[i] = 0.0;
  }

  const auto sqrt_up_11 = sqrt(up_11);
  const auto phi_times_sqrt_up_11 = get(conformal_factor) * sqrt_up_11;

  // TODO : + or -
  r22[7] = 0.5 * e * phi_times_sqrt_up_11;
  r22[8] = up_11;
  r22[9] = up_12;
  r22[10] = up_13;

  r22[11] = 1.0;

  for (size_t i = 12; i < char_fields.size(); i++) {
    r22[i] = 0.0;
  }

  // 23
  for (size_t i = 0; i < char_fields.size(); i++) {
    r23[i] = r22[i];
  }

  // 24
  for (size_t i = 0; i < 6; i++) {
    r24[i] = 0.0;
  }

  // TODO : + or -
  r24[6] = 3.0 * phi_times_sqrt_up_11;
  r24[7] = -4.0 * up_11;
  r24[8] = -4.0 * up_12;
  r24[9] = -4.0 * up_13;

  r24[10] = -3.0;

  for (size_t i = 11; i < 32; i++) {
    r24[i] = 0.0;
  }

  r24[32] = 1.0;

  r24[33] = 0.0;
  r24[34] = 0.0;

  // 25
  const auto one_over_sqrt_up_11 = 1.0 / sqrt_up_11;

  // TODO : + or -
  r25[0] = 2.0 * get(conformal_factor) * up_12 * one_over_sqrt_up_11;
  r25[1] = phi_times_sqrt_up_11;

  for (size_t i = 2; i < 14; i++) {
    r25[i] = 0.0;
  }

  r25[14] = 2.0 * r5[11];

  r25[15] = 1.0;

  for (size_t i = 16; i < char_fields.size(); i++) {
    r25[i] = 0.0;
  }

  // 26
  // TODO : + or -
  r26[0] = 2.0 * get(conformal_factor) * up_13 * one_over_sqrt_up_11;

  r26[1] = 0.0;

  r26[2] = phi_times_sqrt_up_11;

  for (size_t i = 3; i < 14; i++) {
    r26[i] = 0.0;
  }

  r26[14] = 2.0 * r6[11];

  r26[15] = 0.0;

  r26[16] = 1.0;

  for (size_t i = 17; i < char_fields.size(); i++) {
    r26[i] = 0.0;
  }

  // 27
  // TODO : + or -
  r27[0] = get(conformal_factor) * up_22 * one_over_sqrt_up_11;

  r27[1] = 0.0;
  r27[2] = 0.0;

  r27[3] = phi_times_sqrt_up_11;

  for (size_t i = 4; i < 14; i++) {
    r27[i] = 0.0;
  }

  r27[14] = -r6[10];

  r27[15] = 0.0;
  r27[16] = 0.0;

  r27[17] = 1.0;

  for (size_t i = 18; i < char_fields.size(); i++) {
    r27[i] = 0.0;
  }

  // 28
  // TODO : + or -
  r28[0] = 2.0 * get(conformal_factor) * up_23 * one_over_sqrt_up_11;

  r28[1] = 0.0;
  r28[2] = 0.0;
  r28[3] = 0.0;

  r28[4] = phi_times_sqrt_up_11;

  for (size_t i = 5; i < 14; i++) {
    r28[i] = 0.0;
  }

  r28[14] = 2.0 * r6[9];

  r28[15] = 0.0;
  r28[16] = 0.0;
  r28[17] = 0.0;

  r28[18] = 1.0;

  for (size_t i = 19; i < char_fields.size(); i++) {
    r28[i] = 0.0;
  }

  // 29
  // TODO : + or -
  r29[0] = get(conformal_factor) * up_33 * one_over_sqrt_up_11;

  r29[1] = 0.0;
  r29[2] = 0.0;
  r29[3] = 0.0;
  r29[4] = 0.0;

  r29[5] = phi_times_sqrt_up_11;

  for (size_t i = 6; i < 14; i++) {
    r29[i] = 0.0;
  }

  r29[14] = -r5[9];

  r29[15] = 0.0;
  r29[16] = 0.0;
  r29[17] = 0.0;
  r29[18] = 0.0;

  r29[19] = 1.0;

  for (size_t i = 20; i < char_fields.size(); i++) {
    r29[i] = 0.0;
  }

  // 30
  for (size_t i = 0; i < char_fields.size(); i++) {
    r30[i] = r24[i];
  }

  // 31
  for (size_t i = 0; i < char_fields.size(); i++) {
    r31[i] = r25[i];
  }

  // 32
  for (size_t i = 0; i < char_fields.size(); i++) {
    r32[i] = r26[i];
  }

  // 33
  for (size_t i = 0; i < char_fields.size(); i++) {
    r33[i] = r27[i];
  }

  // 34
  for (size_t i = 0; i < char_fields.size(); i++) {
    r34[i] = r28[i];
  }

  // 35
  for (size_t i = 0; i < char_fields.size(); i++) {
    r35[i] = r29[i];
  }
}

template <size_t Dim, typename Frame>
typename Tags::CharacteristicFields<Dim, Frame>::type characteristic_fields(
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame>& inverse_conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const double cleaning_speed) {
  auto char_fields =
      make_with_value<typename Tags::CharacteristicFields<Dim, Frame>::type>(
          get(lapse), 0.);
  characteristic_fields(make_not_null(&char_fields), conformal_spatial_metric,
                        inverse_conformal_spatial_metric, conformal_factor,
                        lapse, cleaning_speed);
  return char_fields;
}
