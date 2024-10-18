#include <cmath>
#include <migraphx/float_equal.hpp>
#include <migraphx/half.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/bit_cast.hpp>
#include "test.hpp"

#include <array>
#include <map>
#include <set>
#include <iomanip>
#include <sstream>
#include <random>
#include <limits>

template <class T, class U>
bool bit_equal(const T& x, const U& y)
{
    static_assert(sizeof(T) == sizeof(U));
    using type = std::array<char, sizeof(T)>;
    return migraphx::bit_cast<type>(x) == migraphx::bit_cast<type>(y);
}

static const std::map<uint16_t, float> half_lut = {
    {0x0000, 0},
    {0x0058, 0.0000052452087402},
    {0x0079, 0.0000072121620178},
    {0x0097, 0.0000090003013611},
    {0x009e, 0.0000094175338745},
    {0x0125, 0.0000174641609192},
    {0x0167, 0.0000213980674744},
    {0x0196, 0.0000241994857788},
    {0x01c4, 0.0000269412994385},
    {0x01c8, 0.0000271797180176},
    {0x0236, 0.0000337362289429},
    {0x029f, 0.0000399947166443},
    {0x02bf, 0.0000419020652771},
    {0x02d6, 0.0000432729721069},
    {0x03a6, 0.0000556707382202},
    {0x03b7, 0.0000566840171814},
    {0x03d4, 0.0000584125518799},
    {0x03d8, 0.000058650970459},
    {0x03ed, 0.0000599026679993},
    {0x0427, 0.0000633597373962},
    {0x0430, 0.0000638961791992},
    {0x0435, 0.0000641942024231},
    {0x0454, 0.0000660419464111},
    {0x047a, 0.0000683069229126},
    {0x04b6, 0.0000718832015991},
    {0x056a, 0.0000826120376587},
    {0x056f, 0.0000829100608826},
    {0x0584, 0.0000841617584229},
    {0x05a1, 0.0000858902931213},
    {0x05a4, 0.0000860691070557},
    {0x05b8, 0.0000872611999512},
    {0x05bc, 0.0000874996185303},
    {0x0635, 0.0000947117805481},
    {0x0641, 0.0000954270362854},
    {0x0686, 0.0000995397567749},
    {0x0694, 0.0001003742218018},
    {0x06db, 0.0001046061515808},
    {0x0725, 0.0001090168952942},
    {0x0777, 0.0001139044761658},
    {0x07b2, 0.0001174211502075},
    {0x0812, 0.0001242160797119},
    {0x082e, 0.0001275539398193},
    {0x0859, 0.00013267993927},
    {0x0895, 0.0001398324966431},
    {0x08af, 0.0001429319381714},
    {0x08fc, 0.0001521110534668},
    {0x092e, 0.0001580715179443},
    {0x0971, 0.0001660585403442},
    {0x0991, 0.0001698732376099},
    {0x09ca, 0.0001766681671143},
    {0x0a63, 0.0001949071884155},
    {0x0a8e, 0.0002000331878662},
    {0x0a93, 0.000200629234314},
    {0x0b2a, 0.0002186298370361},
    {0x0b3a, 0.0002205371856689},
    {0x0b3c, 0.000220775604248},
    {0x0b4e, 0.00022292137146},
    {0x0bae, 0.0002343654632568},
    {0x0bff, 0.0002440214157104},
    {0x0c08, 0.0002460479736328},
    {0x0c56, 0.0002646446228027},
    {0x0c61, 0.0002672672271729},
    {0x0c70, 0.0002708435058594},
    {0x0c7c, 0.0002737045288086},
    {0x0cd8, 0.0002956390380859},
    {0x0cdd, 0.0002968311309814},
    {0x0d05, 0.0003063678741455},
    {0x0d61, 0.0003283023834229},
    {0x0d85, 0.0003368854522705},
    {0x0d8c, 0.0003385543823242},
    {0x0d90, 0.0003395080566406},
    {0x0d9e, 0.000342845916748},
    {0x0da5, 0.0003445148468018},
    {0x0dda, 0.0003571510314941},
    {0x0dde, 0.0003581047058105},
    {0x0df6, 0.000363826751709},
    {0x0eec, 0.000422477722168},
    {0x0f1c, 0.0004339218139648},
    {0x0f99, 0.0004637241363525},
    {0x0fac, 0.0004682540893555},
    {0x0fb0, 0.0004692077636719},
    {0x0ff5, 0.0004856586456299},
    {0x107f, 0.0005488395690918},
    {0x1096, 0.0005598068237305},
    {0x10c8, 0.0005836486816406},
    {0x10e9, 0.0005993843078613},
    {0x110a, 0.000615119934082},
    {0x118a, 0.000676155090332},
    {0x11b5, 0.0006966590881348},
    {0x1293, 0.0008025169372559},
    {0x133f, 0.0008845329284668},
    {0x1342, 0.0008859634399414},
    {0x1372, 0.0009088516235352},
    {0x13cf, 0.000953197479248},
    {0x140c, 0.0009880065917969},
    {0x1437, 0.0010290145874023},
    {0x14a3, 0.0011320114135742},
    {0x14a6, 0.0011348724365234},
    {0x14b2, 0.0011463165283203},
    {0x14ba, 0.0011539459228516},
    {0x14d9, 0.0011835098266602},
    {0x14da, 0.0011844635009766},
    {0x14e7, 0.0011968612670898},
    {0x14fe, 0.0012187957763672},
    {0x1521, 0.0012521743774414},
    {0x153d, 0.0012788772583008},
    {0x15ad, 0.0013856887817383},
    {0x15fd, 0.0014619827270508},
    {0x1649, 0.0015344619750977},
    {0x1658, 0.0015487670898438},
    {0x168a, 0.0015964508056641},
    {0x169d, 0.0016145706176758},
    {0x16b3, 0.0016355514526367},
    {0x16c9, 0.0016565322875977},
    {0x16d1, 0.0016641616821289},
    {0x16e0, 0.001678466796875},
    {0x170a, 0.0017185211181641},
    {0x176d, 0.0018129348754883},
    {0x185b, 0.0021266937255859},
    {0x185e, 0.0021324157714844},
    {0x187e, 0.0021934509277344},
    {0x18ca, 0.0023384094238281},
    {0x18e9, 0.0023975372314453},
    {0x1901, 0.0024433135986328},
    {0x191e, 0.0024986267089844},
    {0x1963, 0.0026302337646484},
    {0x199f, 0.0027446746826172},
    {0x19b2, 0.0027809143066406},
    {0x19d4, 0.0028457641601562},
    {0x1a31, 0.0030231475830078},
    {0x1a4a, 0.0030708312988281},
    {0x1a7a, 0.0031623840332031},
    {0x1ace, 0.0033226013183594},
    {0x1b03, 0.0034236907958984},
    {0x1b22, 0.0034828186035156},
    {0x1d49, 0.0051612854003906},
    {0x1d5a, 0.0052261352539062},
    {0x1d6c, 0.0052947998046875},
    {0x1e02, 0.0058670043945312},
    {0x1e19, 0.0059547424316406},
    {0x1e4c, 0.0061492919921875},
    {0x1eb3, 0.0065422058105469},
    {0x1f32, 0.0070266723632812},
    {0x1f36, 0.0070419311523438},
    {0x1f41, 0.0070838928222656},
    {0x1f7a, 0.0073013305664062},
    {0x1f8d, 0.0073738098144531},
    {0x200b, 0.0078964233398438},
    {0x205f, 0.0085372924804688},
    {0x2060, 0.008544921875},
    {0x2067, 0.0085983276367188},
    {0x20e2, 0.0095367431640625},
    {0x2164, 0.010528564453125},
    {0x22a4, 0.012969970703125},
    {0x22b4, 0.013092041015625},
    {0x22f2, 0.0135650634765625},
    {0x230c, 0.013763427734375},
    {0x2314, 0.013824462890625},
    {0x2341, 0.0141677856445312},
    {0x2356, 0.0143280029296875},
    {0x236e, 0.0145111083984375},
    {0x2371, 0.0145339965820312},
    {0x23cd, 0.0152359008789062},
    {0x2405, 0.0157012939453125},
    {0x24a2, 0.018096923828125},
    {0x24ba, 0.018463134765625},
    {0x24e7, 0.0191497802734375},
    {0x266c, 0.02508544921875},
    {0x26a2, 0.025909423828125},
    {0x26cc, 0.02655029296875},
    {0x26f0, 0.027099609375},
    {0x271e, 0.027801513671875},
    {0x2798, 0.0296630859375},
    {0x287d, 0.035064697265625},
    {0x28a2, 0.03619384765625},
    {0x28ca, 0.03741455078125},
    {0x2933, 0.040618896484375},
    {0x298d, 0.043365478515625},
    {0x299e, 0.04388427734375},
    {0x29c0, 0.044921875},
    {0x29c2, 0.04498291015625},
    {0x29cf, 0.045379638671875},
    {0x29fa, 0.04669189453125},
    {0x2a06, 0.04705810546875},
    {0x2aa5, 0.051910400390625},
    {0x2bcb, 0.060882568359375},
    {0x2c18, 0.06396484375},
    {0x2c65, 0.06866455078125},
    {0x2c66, 0.0687255859375},
    {0x2c93, 0.07147216796875},
    {0x2d24, 0.080322265625},
    {0x2d35, 0.08135986328125},
    {0x2d4c, 0.082763671875},
    {0x2db7, 0.08929443359375},
    {0x2dec, 0.092529296875},
    {0x2e31, 0.09674072265625},
    {0x2ec9, 0.10601806640625},
    {0x2f85, 0.11749267578125},
    {0x2f94, 0.118408203125},
    {0x302b, 0.1302490234375},
    {0x3094, 0.14306640625},
    {0x3096, 0.143310546875},
    {0x30ae, 0.146240234375},
    {0x30b9, 0.1475830078125},
    {0x310c, 0.15771484375},
    {0x31bd, 0.1793212890625},
    {0x3213, 0.1898193359375},
    {0x325b, 0.1986083984375},
    {0x32aa, 0.208251953125},
    {0x32c0, 0.2109375},
    {0x32d7, 0.2137451171875},
    {0x3391, 0.2364501953125},
    {0x340d, 0.253173828125},
    {0x343d, 0.264892578125},
    {0x3566, 0.33740234375},
    {0x35e6, 0.36865234375},
    {0x35f4, 0.3720703125},
    {0x363b, 0.389404296875},
    {0x363e, 0.39013671875},
    {0x3650, 0.39453125},
    {0x3698, 0.412109375},
    {0x36e7, 0.431396484375},
    {0x36fe, 0.43701171875},
    {0x374a, 0.45556640625},
    {0x3760, 0.4609375},
    {0x3761, 0.461181640625},
    {0x379e, 0.47607421875},
    {0x37cc, 0.4873046875},
    {0x37fd, 0.499267578125},
    {0x3828, 0.51953125},
    {0x3841, 0.53173828125},
    {0x3877, 0.55810546875},
    {0x38a4, 0.580078125},
    {0x38d3, 0.60302734375},
    {0x39b2, 0.7119140625},
    {0x3a60, 0.796875},
    {0x3aa3, 0.82958984375},
    {0x3aa6, 0.8310546875},
    {0x3ac9, 0.84814453125},
    {0x3acf, 0.85107421875},
    {0x3b14, 0.884765625},
    {0x3b42, 0.9072265625},
    {0x3b5c, 0.919921875},
    {0x3bde, 0.9833984375},
    {0x3c67, 1.1005859375},
    {0x3cb5, 1.1767578125},
    {0x3cca, 1.197265625},
    {0x3cdd, 1.2158203125},
    {0x3cfc, 1.24609375},
    {0x3d1f, 1.2802734375},
    {0x3e0c, 1.51171875},
    {0x3e1c, 1.52734375},
    {0x3e5b, 1.5888671875},
    {0x3e7f, 1.6240234375},
    {0x3eae, 1.669921875},
    {0x3efe, 1.748046875},
    {0x3f3e, 1.810546875},
    {0x3f9d, 1.9033203125},
    {0x400a, 2.01953125},
    {0x4070, 2.21875},
    {0x40a0, 2.3125},
    {0x40ce, 2.40234375},
    {0x40e6, 2.44921875},
    {0x410e, 2.52734375},
    {0x4129, 2.580078125},
    {0x4144, 2.6328125},
    {0x41a4, 2.8203125},
    {0x41f3, 2.974609375},
    {0x42f1, 3.470703125},
    {0x438f, 3.779296875},
    {0x43b0, 3.84375},
    {0x43c3, 3.880859375},
    {0x43de, 3.93359375},
    {0x4483, 4.51171875},
    {0x44f8, 4.96875},
    {0x4505, 5.01953125},
    {0x45dd, 5.86328125},
    {0x45f3, 5.94921875},
    {0x460e, 6.0546875},
    {0x46ce, 6.8046875},
    {0x4704, 7.015625},
    {0x471a, 7.1015625},
    {0x475e, 7.3671875},
    {0x4761, 7.37890625},
    {0x479f, 7.62109375},
    {0x47ca, 7.7890625},
    {0x47db, 7.85546875},
    {0x47fc, 7.984375},
    {0x481e, 8.234375},
    {0x4839, 8.4453125},
    {0x483d, 8.4765625},
    {0x48ac, 9.34375},
    {0x48da, 9.703125},
    {0x4919, 10.1953125},
    {0x4950, 10.625},
    {0x4987, 11.0546875},
    {0x49bb, 11.4609375},
    {0x4a14, 12.15625},
    {0x4a92, 13.140625},
    {0x4b25, 14.2890625},
    {0x4b81, 15.0078125},
    {0x4b99, 15.1953125},
    {0x4bbe, 15.484375},
    {0x4bf8, 15.9375},
    {0x4c1f, 16.484375},
    {0x4c49, 17.140625},
    {0x4d21, 20.515625},
    {0x4d4a, 21.15625},
    {0x4d51, 21.265625},
    {0x4de2, 23.53125},
    {0x4e05, 24.078125},
    {0x4ea3, 26.546875},
    {0x4eb0, 26.75},
    {0x4f0e, 28.21875},
    {0x4f4a, 29.15625},
    {0x4f6b, 29.671875},
    {0x4fa6, 30.59375},
    {0x4fae, 30.71875},
    {0x4ff6, 31.84375},
    {0x503c, 33.875},
    {0x50e4, 39.125},
    {0x514e, 42.4375},
    {0x516b, 43.34375},
    {0x51d3, 46.59375},
    {0x5213, 48.59375},
    {0x526e, 51.4375},
    {0x52a6, 53.1875},
    {0x52b4, 53.625},
    {0x52b6, 53.6875},
    {0x52bc, 53.875},
    {0x5300, 56},
    {0x5389, 60.28125},
    {0x5406, 64.375},
    {0x5498, 73.5},
    {0x54bd, 75.8125},
    {0x54cf, 76.9375},
    {0x5502, 80.125},
    {0x558e, 88.875},
    {0x5597, 89.4375},
    {0x55eb, 94.6875},
    {0x55f6, 95.375},
    {0x5629, 98.5625},
    {0x562b, 98.6875},
    {0x5635, 99.3125},
    {0x564e, 100.875},
    {0x5671, 103.0625},
    {0x5681, 104.0625},
    {0x56d1, 109.0625},
    {0x571c, 113.75},
    {0x5756, 117.375},
    {0x5790, 121},
    {0x57fd, 127.8125},
    {0x582d, 133.625},
    {0x5869, 141.125},
    {0x58ab, 149.375},
    {0x58ad, 149.625},
    {0x58c9, 153.125},
    {0x58f7, 158.875},
    {0x5904, 160.5},
    {0x59c2, 184.25},
    {0x59e6, 188.75},
    {0x5a88, 209},
    {0x5ada, 219.25},
    {0x5aef, 221.875},
    {0x5af5, 222.625},
    {0x5b7f, 239.875},
    {0x5ba4, 244.5},
    {0x5c08, 258},
    {0x5cbf, 303.75},
    {0x5d4d, 339.25},
    {0x5dc2, 368.5},
    {0x5dc4, 369},
    {0x5e31, 396.25},
    {0x5e38, 398},
    {0x5e7c, 415},
    {0x5e8d, 419.25},
    {0x5ead, 427.25},
    {0x5eb4, 429},
    {0x5ec0, 432},
    {0x5eef, 443.75},
    {0x5f04, 449},
    {0x5f41, 464.25},
    {0x5f58, 470},
    {0x5f61, 472.25},
    {0x5f77, 477.75},
    {0x5f7b, 478.75},
    {0x6029, 532.5},
    {0x6046, 547},
    {0x6055, 554.5},
    {0x60a8, 596},
    {0x60d7, 619.5},
    {0x6139, 668.5},
    {0x6167, 691.5},
    {0x61b5, 730.5},
    {0x61c0, 736},
    {0x61e6, 755},
    {0x625b, 813.5},
    {0x62c4, 866},
    {0x62fd, 894.5},
    {0x62fe, 895},
    {0x6332, 921},
    {0x636a, 949},
    {0x6374, 954},
    {0x6376, 955},
    {0x639f, 975.5},
    {0x63d6, 1003},
    {0x6417, 1047},
    {0x642e, 1070},
    {0x6431, 1073},
    {0x644f, 1103},
    {0x6459, 1113},
    {0x645b, 1115},
    {0x6480, 1152},
    {0x648d, 1165},
    {0x649f, 1183},
    {0x64bb, 1211},
    {0x6516, 1302},
    {0x6571, 1393},
    {0x6585, 1413},
    {0x65aa, 1450},
    {0x660c, 1548},
    {0x6694, 1684},
    {0x66d0, 1744},
    {0x6721, 1825},
    {0x672d, 1837},
    {0x6734, 1844},
    {0x6766, 1894},
    {0x6773, 1907},
    {0x677d, 1917},
    {0x679a, 1946},
    {0x690f, 2590},
    {0x6934, 2664},
    {0x6955, 2730},
    {0x697d, 2810},
    {0x698e, 2844},
    {0x6a3a, 3188},
    {0x6a63, 3270},
    {0x6a67, 3278},
    {0x6a7c, 3320},
    {0x6a87, 3342},
    {0x6b07, 3598},
    {0x6b11, 3618},
    {0x6b36, 3692},
    {0x6b3c, 3704},
    {0x6b75, 3818},
    {0x6b88, 3856},
    {0x6be6, 4044},
    {0x6bee, 4060},
    {0x6c62, 4488},
    {0x6c8b, 4652},
    {0x6d30, 5312},
    {0x6d48, 5408},
    {0x6ddd, 6004},
    {0x6de9, 6052},
    {0x6e39, 6372},
    {0x6e7e, 6648},
    {0x6ea5, 6804},
    {0x6ec5, 6932},
    {0x6ee1, 7044},
    {0x6ef1, 7108},
    {0x6fa2, 7816},
    {0x6fbc, 7920},
    {0x704c, 8800},
    {0x7083, 9240},
    {0x7108, 10304},
    {0x7115, 10408},
    {0x7128, 10560},
    {0x71af, 11640},
    {0x7222, 12560},
    {0x7228, 12608},
    {0x72a5, 13608},
    {0x72e0, 14080},
    {0x72e6, 14128},
    {0x731e, 14576},
    {0x7377, 15288},
    {0x741d, 16848},
    {0x7423, 16944},
    {0x7424, 16960},
    {0x7466, 18016},
    {0x74b0, 19200},
    {0x74ce, 19680},
    {0x74f0, 20224},
    {0x754b, 21680},
    {0x7575, 22352},
    {0x7594, 22848},
    {0x75b1, 23312},
    {0x7614, 24896},
    {0x7618, 24960},
    {0x7631, 25360},
    {0x7660, 26112},
    {0x76c8, 27776},
    {0x7773, 30512},
    {0x77af, 31472},
    {0x77b9, 31632},
    {0x77de, 32224},
    {0x7844, 34944},
    {0x78d2, 39488},
    {0x7924, 42112},
    {0x793b, 42848},
    {0x79db, 47968},
    {0x7a0f, 49632},
    {0x7a1a, 49984},
    {0x7a6c, 52608},
    {0x7a99, 54048},
    {0x7ada, 56128},
    {0x7b0f, 57824},
    {0x7b15, 58016},
    {0x7b41, 59424},
    {0x7b51, 59936},
    {0x7b9c, 62336},
    {0x7ba3, 62560},
    {0x7c00, std::numeric_limits<float>::infinity()},
    {0x7c05, std::numeric_limits<float>::quiet_NaN()},
    {0x7c0e, std::numeric_limits<float>::quiet_NaN()},
    {0x7c3e, std::numeric_limits<float>::quiet_NaN()},
    {0x7c4e, std::numeric_limits<float>::quiet_NaN()},
    {0x7c55, std::numeric_limits<float>::quiet_NaN()},
    {0x7c58, std::numeric_limits<float>::quiet_NaN()},
    {0x7c66, std::numeric_limits<float>::quiet_NaN()},
    {0x7cc9, std::numeric_limits<float>::quiet_NaN()},
    {0x7cd8, std::numeric_limits<float>::quiet_NaN()},
    {0x7d2d, std::numeric_limits<float>::quiet_NaN()},
    {0x7d60, std::numeric_limits<float>::quiet_NaN()},
    {0x7d79, std::numeric_limits<float>::quiet_NaN()},
    {0x7dc7, std::numeric_limits<float>::quiet_NaN()},
    {0x7dcf, std::numeric_limits<float>::quiet_NaN()},
    {0x7dd8, std::numeric_limits<float>::quiet_NaN()},
    {0x7dfb, std::numeric_limits<float>::quiet_NaN()},
    {0x7e0f, std::numeric_limits<float>::quiet_NaN()},
    {0x7e56, std::numeric_limits<float>::quiet_NaN()},
    {0x7e89, std::numeric_limits<float>::quiet_NaN()},
    {0x7e9c, std::numeric_limits<float>::quiet_NaN()},
    {0x7eb2, std::numeric_limits<float>::quiet_NaN()},
    {0x7ec3, std::numeric_limits<float>::quiet_NaN()},
    {0x7ef9, std::numeric_limits<float>::quiet_NaN()},
    {0x7f36, std::numeric_limits<float>::quiet_NaN()},
    {0x8040, -0.0000038146972656},
    {0x8101, -0.0000153183937073},
    {0x813d, -0.0000188946723938},
    {0x81a8, -0.0000252723693848},
    {0x81bc, -0.0000264644622803},
    {0x81c2, -0.0000268220901489},
    {0x8259, -0.00003582239151},
    {0x8330, -0.0000486373901367},
    {0x8366, -0.0000518560409546},
    {0x8392, -0.0000544786453247},
    {0x83e4, -0.0000593662261963},
    {0x83ee, -0.000059962272644},
    {0x8402, -0.0000611543655396},
    {0x845e, -0.0000666379928589},
    {0x84ac, -0.0000712871551514},
    {0x84b1, -0.0000715851783752},
    {0x84fb, -0.0000759959220886},
    {0x8546, -0.0000804662704468},
    {0x856f, -0.0000829100608826},
    {0x85b5, -0.0000870823860168},
    {0x8638, -0.0000948905944824},
    {0x8656, -0.0000966787338257},
    {0x86b9, -0.0001025795936584},
    {0x86ba, -0.0001026391983032},
    {0x86fe, -0.0001066923141479},
    {0x8731, -0.0001097321510315},
    {0x8740, -0.0001106262207031},
    {0x8793, -0.0001155734062195},
    {0x87bd, -0.0001180768013},
    {0x87f1, -0.0001211762428284},
    {0x87f4, -0.0001213550567627},
    {0x8809, -0.000123143196106},
    {0x882a, -0.0001270771026611},
    {0x8848, -0.0001306533813477},
    {0x8852, -0.0001318454742432},
    {0x8874, -0.0001358985900879},
    {0x8892, -0.0001394748687744},
    {0x88a7, -0.000141978263855},
    {0x88c8, -0.0001459121704102},
    {0x8927, -0.0001572370529175},
    {0x892a, -0.0001575946807861},
    {0x8989, -0.0001689195632935},
    {0x89b9, -0.0001746416091919},
    {0x8b18, -0.0002164840698242},
    {0x8b4b, -0.0002225637435913},
    {0x8b62, -0.000225305557251},
    {0x8b7f, -0.0002287626266479},
    {0x8bca, -0.0002377033233643},
    {0x8bcf, -0.000238299369812},
    {0x8bff, -0.0002440214157104},
    {0x8c0b, -0.0002467632293701},
    {0x8c55, -0.0002644062042236},
    {0x8c63, -0.0002677440643311},
    {0x8d53, -0.0003249645233154},
    {0x8dba, -0.0003495216369629},
    {0x8e03, -0.0003669261932373},
    {0x8e82, -0.0003972053527832},
    {0x8e9c, -0.0004034042358398},
    {0x8faa, -0.0004677772521973},
    {0x902f, -0.0005106925964355},
    {0x9051, -0.0005269050598145},
    {0x9066, -0.0005369186401367},
    {0x907e, -0.0005483627319336},
    {0x9080, -0.00054931640625},
    {0x908e, -0.0005559921264648},
    {0x9102, -0.0006113052368164},
    {0x91eb, -0.0007224082946777},
    {0x9215, -0.0007424354553223},
    {0x9252, -0.0007715225219727},
    {0x9294, -0.0008029937744141},
    {0x9297, -0.0008044242858887},
    {0x933d, -0.0008835792541504},
    {0x936f, -0.0009074211120605},
    {0x93aa, -0.0009355545043945},
    {0x93f2, -0.0009698867797852},
    {0x941d, -0.0010042190551758},
    {0x945a, -0.0010623931884766},
    {0x94ad, -0.0011415481567383},
    {0x94d2, -0.0011768341064453},
    {0x951c, -0.0012474060058594},
    {0x9520, -0.001251220703125},
    {0x952f, -0.0012655258178711},
    {0x953f, -0.0012807846069336},
    {0x9549, -0.0012903213500977},
    {0x95c6, -0.0014095306396484},
    {0x9602, -0.0014667510986328},
    {0x969b, -0.001612663269043},
    {0x96fa, -0.0017032623291016},
    {0x977d, -0.0018281936645508},
    {0x97c3, -0.0018949508666992},
    {0x97c6, -0.0018978118896484},
    {0x97db, -0.001917839050293},
    {0x97f9, -0.0019464492797852},
    {0x983f, -0.0020732879638672},
    {0x984e, -0.0021018981933594},
    {0x985a, -0.0021247863769531},
    {0x988c, -0.0022201538085938},
    {0x990d, -0.0024662017822266},
    {0x9958, -0.0026092529296875},
    {0x9971, -0.0026569366455078},
    {0x9a4e, -0.0030784606933594},
    {0x9a8f, -0.0032024383544922},
    {0x9abe, -0.0032920837402344},
    {0x9ace, -0.0033226013183594},
    {0x9b1e, -0.0034751892089844},
    {0x9b3e, -0.0035362243652344},
    {0x9b77, -0.0036449432373047},
    {0x9b89, -0.0036792755126953},
    {0x9b90, -0.003692626953125},
    {0x9bec, -0.0038681030273438},
    {0x9c03, -0.0039176940917969},
    {0x9c75, -0.0043525695800781},
    {0x9d6c, -0.0052947998046875},
    {0x9d74, -0.0053253173828125},
    {0x9da7, -0.0055198669433594},
    {0x9e73, -0.0062980651855469},
    {0x9e94, -0.0064239501953125},
    {0x9f17, -0.0069236755371094},
    {0x9f3a, -0.0070571899414062},
    {0x9f6c, -0.0072479248046875},
    {0x9f89, -0.0073585510253906},
    {0x9fbd, -0.0075569152832031},
    {0xa003, -0.0078353881835938},
    {0xa014, -0.007965087890625},
    {0xa019, -0.0080032348632812},
    {0xa01d, -0.0080337524414062},
    {0xa090, -0.0089111328125},
    {0xa1cf, -0.0113449096679688},
    {0xa1dd, -0.0114517211914062},
    {0xa249, -0.0122756958007812},
    {0xa26d, -0.0125503540039062},
    {0xa288, -0.01275634765625},
    {0xa2fb, -0.0136337280273438},
    {0xa390, -0.0147705078125},
    {0xa3b3, -0.0150375366210938},
    {0xa3ed, -0.0154800415039062},
    {0xa434, -0.01641845703125},
    {0xa476, -0.017425537109375},
    {0xa571, -0.0212554931640625},
    {0xa57d, -0.0214385986328125},
    {0xa597, -0.0218353271484375},
    {0xa5d1, -0.0227203369140625},
    {0xa5f9, -0.0233306884765625},
    {0xa680, -0.025390625},
    {0xa6e3, -0.0269012451171875},
    {0xa6f0, -0.027099609375},
    {0xa72d, -0.0280303955078125},
    {0xa77e, -0.029266357421875},
    {0xa7d0, -0.030517578125},
    {0xa7ee, -0.030975341796875},
    {0xa7f3, -0.0310516357421875},
    {0xa80c, -0.0316162109375},
    {0xa827, -0.032440185546875},
    {0xa89f, -0.036102294921875},
    {0xa8a0, -0.0361328125},
    {0xa8a5, -0.036285400390625},
    {0xa948, -0.041259765625},
    {0xaa0c, -0.0472412109375},
    {0xaa16, -0.04754638671875},
    {0xaa9a, -0.05157470703125},
    {0xaaeb, -0.054046630859375},
    {0xab5c, -0.0574951171875},
    {0xac7e, -0.0701904296875},
    {0xad33, -0.08123779296875},
    {0xad37, -0.08148193359375},
    {0xad90, -0.0869140625},
    {0xada0, -0.087890625},
    {0xade5, -0.09210205078125},
    {0xadf8, -0.09326171875},
    {0xae02, -0.0938720703125},
    {0xae04, -0.093994140625},
    {0xae4f, -0.09857177734375},
    {0xae63, -0.09979248046875},
    {0xaebe, -0.1053466796875},
    {0xaee1, -0.10748291015625},
    {0xaef9, -0.10894775390625},
    {0xaf0b, -0.11004638671875},
    {0xaf78, -0.11669921875},
    {0xaf7d, -0.11700439453125},
    {0xaf7f, -0.11712646484375},
    {0xaf8c, -0.117919921875},
    {0xafcb, -0.12176513671875},
    {0xb06b, -0.1380615234375},
    {0xb07b, -0.1400146484375},
    {0xb088, -0.1416015625},
    {0xb0b2, -0.146728515625},
    {0xb0ed, -0.1539306640625},
    {0xb0f9, -0.1553955078125},
    {0xb16c, -0.16943359375},
    {0xb189, -0.1729736328125},
    {0xb1c5, -0.1802978515625},
    {0xb1f7, -0.1864013671875},
    {0xb22d, -0.1929931640625},
    {0xb23c, -0.19482421875},
    {0xb258, -0.1982421875},
    {0xb2c7, -0.2117919921875},
    {0xb2de, -0.214599609375},
    {0xb2e1, -0.2149658203125},
    {0xb317, -0.2215576171875},
    {0xb31d, -0.2222900390625},
    {0xb3ef, -0.2479248046875},
    {0xb3f8, -0.2490234375},
    {0xb45a, -0.27197265625},
    {0xb548, -0.330078125},
    {0xb5d8, -0.365234375},
    {0xb64e, -0.39404296875},
    {0xb69f, -0.413818359375},
    {0xb6e6, -0.43115234375},
    {0xb6ed, -0.432861328125},
    {0xb6f7, -0.435302734375},
    {0xb79a, -0.47509765625},
    {0xb7b6, -0.48193359375},
    {0xb7ee, -0.49560546875},
    {0xb856, -0.5419921875},
    {0xb8c0, -0.59375},
    {0xb96f, -0.67919921875},
    {0xb9a5, -0.70556640625},
    {0xba1e, -0.7646484375},
    {0xba2d, -0.77197265625},
    {0xba48, -0.78515625},
    {0xba65, -0.79931640625},
    {0xbaaf, -0.83544921875},
    {0xbab0, -0.8359375},
    {0xbb12, -0.8837890625},
    {0xbb35, -0.90087890625},
    {0xbb47, -0.90966796875},
    {0xbb97, -0.94873046875},
    {0xbba3, -0.95458984375},
    {0xbbcb, -0.97412109375},
    {0xbbe8, -0.98828125},
    {0xbbee, -0.9912109375},
    {0xbd03, -1.2529296875},
    {0xbd4b, -1.3232421875},
    {0xbd4c, -1.32421875},
    {0xbd8a, -1.384765625},
    {0xbdb6, -1.427734375},
    {0xbde1, -1.4697265625},
    {0xbe04, -1.50390625},
    {0xbe50, -1.578125},
    {0xbe54, -1.58203125},
    {0xbe6a, -1.603515625},
    {0xbf31, -1.7978515625},
    {0xbf87, -1.8818359375},
    {0xbfa2, -1.908203125},
    {0xc016, -2.04296875},
    {0xc074, -2.2265625},
    {0xc0ca, -2.39453125},
    {0xc100, -2.5},
    {0xc1b7, -2.857421875},
    {0xc1b9, -2.861328125},
    {0xc1d3, -2.912109375},
    {0xc23f, -3.123046875},
    {0xc2d5, -3.416015625},
    {0xc32f, -3.591796875},
    {0xc3e3, -3.943359375},
    {0xc412, -4.0703125},
    {0xc49a, -4.6015625},
    {0xc4ca, -4.7890625},
    {0xc4cf, -4.80859375},
    {0xc523, -5.13671875},
    {0xc55d, -5.36328125},
    {0xc5aa, -5.6640625},
    {0xc604, -6.015625},
    {0xc61b, -6.10546875},
    {0xc642, -6.2578125},
    {0xc68b, -6.54296875},
    {0xc69e, -6.6171875},
    {0xc6b0, -6.6875},
    {0xc6ca, -6.7890625},
    {0xc71e, -7.1171875},
    {0xc721, -7.12890625},
    {0xc73b, -7.23046875},
    {0xc7d4, -7.828125},
    {0xc831, -8.3828125},
    {0xc89a, -9.203125},
    {0xc8be, -9.484375},
    {0xc8dc, -9.71875},
    {0xc8e4, -9.78125},
    {0xc8fa, -9.953125},
    {0xc8fe, -9.984375},
    {0xc969, -10.8203125},
    {0xca0f, -12.1171875},
    {0xca1a, -12.203125},
    {0xca6f, -12.8671875},
    {0xca7b, -12.9609375},
    {0xca8f, -13.1171875},
    {0xcaca, -13.578125},
    {0xcafd, -13.9765625},
    {0xcb05, -14.0390625},
    {0xcb6b, -14.8359375},
    {0xcbaf, -15.3671875},
    {0xcbb4, -15.40625},
    {0xcbdf, -15.7421875},
    {0xcc2d, -16.703125},
    {0xcc74, -17.8125},
    {0xccac, -18.6875},
    {0xcd11, -20.265625},
    {0xce04, -24.0625},
    {0xce0f, -24.234375},
    {0xceaf, -26.734375},
    {0xceb8, -26.875},
    {0xcf36, -28.84375},
    {0xcfad, -30.703125},
    {0xd019, -32.78125},
    {0xd08d, -36.40625},
    {0xd115, -40.65625},
    {0xd119, -40.78125},
    {0xd128, -41.25},
    {0xd1a4, -45.125},
    {0xd1b7, -45.71875},
    {0xd1b8, -45.75},
    {0xd203, -48.09375},
    {0xd20a, -48.3125},
    {0xd28b, -52.34375},
    {0xd2ac, -53.375},
    {0xd2ae, -53.4375},
    {0xd2c5, -54.15625},
    {0xd2f2, -55.5625},
    {0xd326, -57.1875},
    {0xd337, -57.71875},
    {0xd343, -58.09375},
    {0xd34e, -58.4375},
    {0xd40c, -64.75},
    {0xd43b, -67.6875},
    {0xd45a, -69.625},
    {0xd464, -70.25},
    {0xd4c3, -76.1875},
    {0xd505, -80.3125},
    {0xd52d, -82.8125},
    {0xd5cf, -92.9375},
    {0xd5f0, -95},
    {0xd607, -96.4375},
    {0xd635, -99.3125},
    {0xd63d, -99.8125},
    {0xd644, -100.25},
    {0xd658, -101.5},
    {0xd789, -120.5625},
    {0xd863, -140.375},
    {0xd866, -140.75},
    {0xd884, -144.5},
    {0xd88d, -145.625},
    {0xd89b, -147.375},
    {0xd8da, -155.25},
    {0xd93b, -167.375},
    {0xd982, -176.25},
    {0xd995, -178.625},
    {0xd99d, -179.625},
    {0xd9cf, -185.875},
    {0xdaaf, -213.875},
    {0xdabd, -215.625},
    {0xdb54, -234.5},
    {0xdc10, -260},
    {0xdca1, -296.25},
    {0xdd0a, -322.5},
    {0xdd56, -341.5},
    {0xddcf, -371.75},
    {0xde04, -385},
    {0xde0d, -387.25},
    {0xde3d, -399.25},
    {0xde4f, -403.75},
    {0xde66, -409.5},
    {0xdeae, -427.5},
    {0xdf52, -468.5},
    {0xdf63, -472.75},
    {0xdf6a, -474.5},
    {0xdf77, -477.75},
    {0xdf7b, -478.75},
    {0xdfc5, -497.25},
    {0xdfcf, -499.75},
    {0xdfd2, -500.5},
    {0xdfd8, -502},
    {0xdfe1, -504.25},
    {0xe022, -529},
    {0xe046, -547},
    {0xe092, -585},
    {0xe0b0, -600},
    {0xe0be, -607},
    {0xe0f4, -634},
    {0xe11b, -653.5},
    {0xe19c, -718},
    {0xe213, -777.5},
    {0xe232, -793},
    {0xe25b, -813.5},
    {0xe262, -817},
    {0xe279, -828.5},
    {0xe2cc, -870},
    {0xe2da, -877},
    {0xe326, -915},
    {0xe330, -920},
    {0xe3c3, -993.5},
    {0xe3cc, -998},
    {0xe566, -1382},
    {0xe57e, -1406},
    {0xe5c8, -1480},
    {0xe609, -1545},
    {0xe628, -1576},
    {0xe663, -1635},
    {0xe6ac, -1708},
    {0xe710, -1808},
    {0xe77f, -1919},
    {0xe7e7, -2023},
    {0xe868, -2256},
    {0xe885, -2314},
    {0xe8ea, -2516},
    {0xe919, -2610},
    {0xe92c, -2648},
    {0xea60, -3264},
    {0xeac1, -3458},
    {0xeacb, -3478},
    {0xeb22, -3652},
    {0xeb2c, -3672},
    {0xeb59, -3762},
    {0xeba5, -3914},
    {0xec53, -4428},
    {0xec97, -4700},
    {0xed16, -5208},
    {0xed4a, -5416},
    {0xed69, -5540},
    {0xee14, -6224},
    {0xee59, -6500},
    {0xee8a, -6696},
    {0xee93, -6732},
    {0xeed7, -7004},
    {0xef0b, -7212},
    {0xef59, -7524},
    {0xef61, -7556},
    {0xef67, -7580},
    {0xefb6, -7896},
    {0xf03a, -8656},
    {0xf04e, -8816},
    {0xf05f, -8952},
    {0xf09f, -9464},
    {0xf0c0, -9728},
    {0xf173, -11160},
    {0xf1d7, -11960},
    {0xf225, -12584},
    {0xf2ca, -13904},
    {0xf2d8, -14016},
    {0xf2e5, -14120},
    {0xf317, -14520},
    {0xf35d, -15080},
    {0xf3bd, -15848},
    {0xf3d3, -16024},
    {0xf3e6, -16176},
    {0xf3fb, -16344},
    {0xf477, -18288},
    {0xf4e0, -19968},
    {0xf4e5, -20048},
    {0xf50b, -20656},
    {0xf5a2, -23072},
    {0xf5c1, -23568},
    {0xf634, -25408},
    {0xf651, -25872},
    {0xf68a, -26784},
    {0xf69c, -27072},
    {0xf6ce, -27872},
    {0xf816, -33472},
    {0xf849, -35104},
    {0xf869, -36128},
    {0xf878, -36608},
    {0xf8cf, -39392},
    {0xf90a, -41280},
    {0xf916, -41664},
    {0xf91e, -41920},
    {0xf9c1, -47136},
    {0xfa0a, -49472},
    {0xfa11, -49696},
    {0xfa1d, -50080},
    {0xfa51, -51744},
    {0xfa86, -53440},
    {0xfaac, -54656},
    {0xfb95, -62112},
    {0xfbd1, -64032},
    {0xfbe0, -64512},
    {0xfbf5, -65184},
    {0xfc00, -std::numeric_limits<float>::infinity()},
    {0xfca5, std::numeric_limits<float>::quiet_NaN()},
    {0xfcb9, std::numeric_limits<float>::quiet_NaN()},
    {0xfcc6, std::numeric_limits<float>::quiet_NaN()},
    {0xfd72, std::numeric_limits<float>::quiet_NaN()},
    {0xfd77, std::numeric_limits<float>::quiet_NaN()},
    {0xfda3, std::numeric_limits<float>::quiet_NaN()},
    {0xfe3e, std::numeric_limits<float>::quiet_NaN()},
    {0xfe89, std::numeric_limits<float>::quiet_NaN()},
    {0xfe91, std::numeric_limits<float>::quiet_NaN()},
    {0xfe93, std::numeric_limits<float>::quiet_NaN()},
    {0xfed1, std::numeric_limits<float>::quiet_NaN()},
    {0xff7a, std::numeric_limits<float>::quiet_NaN()},
    {0xffa3, std::numeric_limits<float>::quiet_NaN()},
};

TEST_CASE(check_half_values)
{
    for(auto [x, f] : half_lut)
    {
        auto h = migraphx::bit_cast<migraphx::half>(x);
        if(std::isnan(f))
        {
            CHECK(std::isnan(h));
        }
        else if(std::isinf(f))
        {
            CHECK(std::isinf(h));
            CHECK((h < 0) == (f < 0));
            CHECK(bit_equal(x, migraphx::half(f)));
        }
        else
        {
            CHECK(migraphx::float_equal(float(h), f));
            CHECK(bit_equal(x, migraphx::half(f)));
        }
    }
}

template <class T>
std::string to_hex(T i)
{
    std::stringstream ss;
    ss << "0x" << std::setfill('0') << std::setw(4) << std::right << std::hex << i;
    return ss.str();
}

template <class T>
std::string to_decimal(T i)
{
    std::stringstream ss;
    ss << std::setprecision(16) << std::fixed << std::noshowpoint << i;
    std::string str           = ss.str();
    std::size_t trailing_zero = str.find_last_not_of('0') + 1;
    str.erase(trailing_zero);
    std::size_t trailing_decimal = str.find_last_not_of('.') + 1;
    str.erase(trailing_decimal);
    return str;
}

float to_float(uint16_t i) { return reinterpret_cast<const migraphx::half&>(i); }

void show(uint16_t i)
{
    float f = to_float(i);
    std::cout << "{ " << to_hex(i) << ", ";
    if(std::isfinite(f))
    {
        std::cout << to_decimal(f);
        // std::cout << std::fixed << f;
        // std::cout << std::setprecision(100) << std::fixed << f;
        // printf("%.10g", f);
    }
    else if(std::isinf(f))
    {
        if(f < 0)
            std::cout << "-";
        std::cout << "std::numeric_limits<float>::infinity()";
    }
    else if(std::isnan(f))
    {
        std::cout << "std::numeric_limits<float>::quiet_NaN()";
    }

    std::cout << " },\n";
}

void sample()
{
    std::set<uint16_t> samples = {0};
    auto ns                    = migraphx::range(1, 65535);
    std::sample(ns.begin(),
                ns.end(),
                std::inserter(samples, samples.begin()),
                1022,
                std::mt19937{std::random_device{}()});
    samples.insert(65536);
    std::copy_if(ns.begin(), ns.end(), std::inserter(samples, samples.begin()), [](auto i) {
        return std::isinf(to_float(i)); // or std::isnan(to_float(i));
    });
    for(auto i : samples)
    {
        // if(std::isnan(to_float(i)) and not std::isinf(to_float(i)))
        // continue;
        show(i);
    }
}

int main(int argc, const char* argv[])
{
    // sample();
    test::run(argc, argv);
}
