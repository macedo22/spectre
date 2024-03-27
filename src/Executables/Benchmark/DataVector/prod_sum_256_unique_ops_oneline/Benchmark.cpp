// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
constexpr size_t seed = 17;
std::mt19937 generator(seed);

void bench_prod_sum_256_unique_ops_oneline(benchmark::State& state) {  // NOLINT
  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataVector used_for_size =
      DataVector(num_grid_points, std::numeric_limits<double>::signaling_NaN());
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const DataVector AA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector AZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector BZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector CZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector DZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ED = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ER = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ES = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ET = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector EZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector FZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector GZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector HZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ID = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector II = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector IZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector JZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector KZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector LZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ME = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ML = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector MZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ND = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector NZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ON = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector OZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector PZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector QZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RT = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector RZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SS = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector ST = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SU = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SV = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SW = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SX = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SY = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector SZ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TA = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TB = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TC = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TD = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TE = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TF = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TG = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TH = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TI = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TJ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TK = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TL = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TM = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TN = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TO = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TP = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TQ = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const DataVector TR = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  for (auto _ : state) {
    DataVector result =
        AA * AB + AC * AD + AE * AF + AG * AH + AI * AJ + AK * AL + AM * AN +
        AO * AP + AQ * AR + AS * AT + AU * AV + AQ * AX + AY * AZ + BA * BB +
        BC * BD + BE * BF + BG * BH + BI * BJ + BK * BL + BM * BN + BO * BP +
        BQ * BR + BS * BT + BU * BV + BQ * BX + BY * BZ + CA * CB + CC * CD +
        CE * CF + CG * CH + CI * CJ + CK * CL + CM * CN + CO * CP + CQ * CR +
        CS * CT + CU * CV + CQ * CX + CY * CZ + DA * DB + DC * DD + DE * DF +
        DG * DH + DI * DJ + DK * DL + DM * DN + DO * DP + DQ * DR + DS * DT +
        DU * DV + DQ * DX + DY * DZ + EA * EB + EC * ED + EE * EF + EG * EH +
        EI * EJ + EK * EL + EM * EN + EO * EP + EQ * ER + ES * ET + EU * EV +
        EQ * EX + EY * EZ + FA * FB + FC * FD + FE * FF + FG * FH + FI * FJ +
        FK * FL + FM * FN + FO * FP + FQ * FR + FS * FT + FU * FV + FQ * FX +
        FY * FZ + GA * GB + GC * GD + GE * GF + GG * GH + GI * GJ + GK * GL +
        GM * GN + GO * GP + GQ * GR + GS * GT + GU * GV + GQ * GX + GY * GZ +
        HA * HB + HC * HD + HE * HF + HG * HH + HI * HJ + HK * HL + HM * HN +
        HO * HP + HQ * HR + HS * HT + HU * HV + HQ * HX + HY * HZ + IA * IB +
        IC * ID + IE * IF + IG * IH + II * IJ + IK * IL + IM * IN + IO * IP +
        IQ * IR + IS * IT + IU * IV + IQ * IX + IY * IZ + JA * JB + JC * JD +
        JE * JF + JG * JH + JI * JJ + JK * JL + JM * JN + JO * JP + JQ * JR +
        JS * JT + JU * JV + JQ * JX + JY * JZ + KA * KB + KC * KD + KE * KF +
        KG * KH + KI * KJ + KK * KL + KM * KN + KO * KP + KQ * KR + KS * KT +
        KU * KV + KQ * KX + KY * KZ + LA * LB + LC * LD + LE * LF + LG * LH +
        LI * LJ + LK * LL + LM * LN + LO * LP + LQ * LR + LS * LT + LU * LV +
        LQ * LX + LY * LZ + MA * MB + MC * MD + ME * MF + MG * MH + MI * MJ +
        MK * ML + MM * MN + MO * MP + MQ * MR + MS * MT + MU * MV + MQ * MX +
        MY * MZ + NA * NB + NC * ND + NE * NF + NG * NH + NI * NJ + NK * NL +
        NM * NN + NO * NP + NQ * NR + NS * NT + NU * NV + NQ * NX + NY * NZ +
        OA * OB + OC * OD + OE * OF + OG * OH + OI * OJ + OK * OL + OM * ON +
        OO * OP + OQ * OR + OS * OT + OU * OV + OQ * OX + OY * OZ + PA * PB +
        PC * PD + PE * PF + PG * PH + PI * PJ + PK * PL + PM * PN + PO * PP +
        PQ * PR + PS * PT + PU * PV + PQ * PX + PY * PZ + QA * QB + QC * QD +
        QE * QF + QG * QH + QI * QJ + QK * QL + QM * QN + QO * QP + QQ * QR +
        QS * QT + QU * QV + QQ * QX + QY * QZ + RA * RB + RC * RD + RE * RF +
        RG * RH + RI * RJ + RK * RL + RM * RN + RO * RP + RQ * RR + RS * RT +
        RU * RV + RQ * RX + RY * RZ + SA * SB + SC * SD + SE * SF + SG * SH +
        SI * SJ + SK * SL + SM * SN + SO * SP + SQ * SR + SS * ST + SU * SV +
        SQ * SX + SY * SZ + TA * TB + TC * TD + TE * TF + TG * TH + TI * TJ +
        TK * TL + TM * TN + TO * TP + TQ * TR;
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
}

// Cases are run with each number of grid points
constexpr std::array<long int, 4> num_grid_point_values = {8, 125, 512, 1000};

BENCHMARK(bench_prod_sum_256_unique_ops_oneline)
    ->Arg(num_grid_point_values[0])
    ->Arg(num_grid_point_values[1])
    ->Arg(num_grid_point_values[2])
    ->Arg(num_grid_point_values[3]);
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
