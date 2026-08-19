#include "voxel_mesher_cubes.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/godot/classes/array_mesh.h"
#include "../../util/godot/classes/base_material_3d.h"
#include "../../util/godot/classes/geometry_2d.h"
#include "../../util/godot/classes/image.h"
#include "../../util/godot/classes/material.h"
#include "../../util/godot/classes/shader_material.h"
#include "../../util/godot/core/packed_arrays.h"
#include "../../util/godot/core/string.h"
#include "../../util/math/conv.h"
#include "../../util/profiling.h"
#include "../../util/string/format.h"

// TODO Binary greedy mesher optimization
// https://www.youtube.com/watch?v=qnGoGq7DWMc

namespace zylann::voxel {

namespace {
// Table of indices for vertices of cube faces
// 2-----3
// |     |
// |     |
// 0-----1
// [axis][front/back][i]
const uint8_t g_indices_lut[3][2][6] = {
	// X
	{
			// Front
			{ 0, 3, 2, 0, 1, 3 },
			// Back
			{ 0, 2, 3, 0, 3, 1 },
	},
	// Y
	{
			// Front
			{ 0, 2, 3, 0, 3, 1 },
			// Back
			{ 0, 3, 2, 0, 1, 3 },
	},
	// Z
	{
			// Front
			{ 0, 3, 2, 0, 1, 3 },
			// Back
			{ 0, 2, 3, 0, 3, 1 },
	}
};

const uint8_t g_face_axes_lut[Vector3iUtil::AXIS_COUNT][2] = {
	// X
	{ Vector3i::AXIS_Y, Vector3i::AXIS_Z },
	// Y
	{ Vector3i::AXIS_X, Vector3i::AXIS_Z },
	// Z
	{ Vector3i::AXIS_X, Vector3i::AXIS_Y }
};

// Not named `Side` because Godot already defines that in global space
enum FaceSide {
	FACE_SIDE_FRONT = 0,
	FACE_SIDE_BACK,
	FACE_SIDE_NONE // Either means there is no face, or it was consumed
};

} // namespace

// Returns the occupancy/owner priority used when deciding which side of a
// voxel boundary owns a face. Air is empty, ordinary intermediate alpha and
// the cutout marker are occupied, blended has higher priority than cutout,
// and opaque has the highest priority. The priority is deliberately shared by
// all three cube-building paths so adjacent CUTOUT/BLENDED voxels have a
// deterministic owner instead of producing duplicate coplanar faces.
inline uint8_t get_alpha_index(Color8 c) {
	if (c.a == 0) {
		return 0;
	}
	if (c.a == 0xff) {
		return 3;
	}
	if (c.a == 0xfd) {
		return 2;
	}
	return 1;
}

// Alpha markers are only used to carry render-class information through the
// native vertex color. Other non-zero intermediate alpha values remain
// CUTOUT for raw/mesher-palette compatibility. Alpha zero never reaches this
// helper for a committed face because get_alpha_index treats it as air.
inline uint8_t get_material_index(Color8 c) {
	return c.a == 0xff ? VoxelMesherCubes::MATERIAL_OPAQUE :
			c.a == 0xfd ? VoxelMesherCubes::MATERIAL_BLENDED : VoxelMesherCubes::MATERIAL_CUTOUT;
}

// Classic Minecraft-style corner AO. The three samples are taken on the
// exposed face plane: two tangent-side voxels and their diagonal corner. A
// full pair of side blockers wins over the diagonal sample, avoiding a bright
// diagonal seam. Values are quantized to 0, 85, 170, or 255 for deterministic
// vertex data and are written to COLOR.g by shader-palette paths.
template <typename Voxel_T, typename Color_F>
inline uint8_t get_vertex_ao(
		const Span<const Voxel_T> voxel_buffer,
		const Vector3i block_size,
		const Vector3i cell_position,
		const unsigned int face_axis,
		const int face_sign,
		const int tangent_x_sign,
		const int tangent_y_sign,
		Color_F color_func
) {
	const auto is_occupied = [&](const Vector3i position) {
		// One voxel of padding is required by the mesher. Treat an absent sample
		// as air as a defensive fallback for direct callers with a short buffer.
		if (position.x < 0 || position.y < 0 || position.z < 0 || position.x >= block_size.x ||
				position.y >= block_size.y || position.z >= block_size.z) {
			return false;
		}
		const unsigned int index = Vector3iUtil::get_zxy_index(position, block_size);
		return get_alpha_index(color_func(voxel_buffer[index])) != 0;
	};

	const unsigned int tangent_x_axis = g_face_axes_lut[face_axis][0];
	const unsigned int tangent_y_axis = g_face_axes_lut[face_axis][1];
	Vector3i sample_base = cell_position;
	sample_base[face_axis] += face_sign;
	Vector3i side_x = sample_base;
	side_x[tangent_x_axis] += tangent_x_sign;
	Vector3i side_y = sample_base;
	side_y[tangent_y_axis] += tangent_y_sign;
	Vector3i corner = side_x;
	corner[tangent_y_axis] += tangent_y_sign;

	const bool side_x_occupied = is_occupied(side_x);
	const bool side_y_occupied = is_occupied(side_y);
	const bool corner_occupied = is_occupied(corner);
	const int occlusion = side_x_occupied && side_y_occupied ? 0 :
			3 - int(side_x_occupied) - int(side_y_occupied) - int(corner_occupied);
	return static_cast<uint8_t>(occlusion * 85);
}

// Meshing always receives one voxel of padding and the face loops below only
// query samples inside that padded buffer. Keep a fast variant without the
// defensive bounds branch in the inner AO loop; the public mesher contract
// already rejects undersized buffers before entering these builders.
template <typename Voxel_T, typename Color_F>
inline uint8_t get_vertex_ao_unchecked(
		const Span<const Voxel_T> voxel_buffer,
		const Vector3i block_size,
		const Vector3i cell_position,
		const unsigned int face_axis,
		const int face_sign,
		const int tangent_x_sign,
		const int tangent_y_sign,
		Color_F color_func
) {
	const unsigned int tangent_x_axis = g_face_axes_lut[face_axis][0];
	const unsigned int tangent_y_axis = g_face_axes_lut[face_axis][1];
	Vector3i sample_base = cell_position;
	sample_base[face_axis] += face_sign;
	Vector3i side_x = sample_base;
	side_x[tangent_x_axis] += tangent_x_sign;
	Vector3i side_y = sample_base;
	side_y[tangent_y_axis] += tangent_y_sign;
	Vector3i corner = side_x;
	corner[tangent_y_axis] += tangent_y_sign;

	const bool side_x_occupied = get_alpha_index(
			color_func(voxel_buffer[Vector3iUtil::get_zxy_index(side_x, block_size)])) != 0;
	const bool side_y_occupied = get_alpha_index(
			color_func(voxel_buffer[Vector3iUtil::get_zxy_index(side_y, block_size)])) != 0;
	const bool corner_occupied = get_alpha_index(
			color_func(voxel_buffer[Vector3iUtil::get_zxy_index(corner, block_size)])) != 0;
	const int occlusion = side_x_occupied && side_y_occupied ? 0 :
			3 - int(side_x_occupied) - int(side_y_occupied) - int(corner_occupied);
	return static_cast<uint8_t>(occlusion * 85);
}

// The four corners of one quad share the two tangent-side samples. Gather the
// eight unique occupancy samples once instead of repeating twelve color/palette
// lookups through four independent AO calls.
template <typename Voxel_T, typename Color_F>
inline void fill_vertex_ao_unchecked(
		const Span<const Voxel_T> voxel_buffer,
		const Vector3i block_size,
		const Vector3i cell_position,
		const unsigned int face_axis,
		const int face_sign,
		Color_F color_func,
		uint8_t (&out_ao)[4]
) {
	const unsigned int tangent_x_axis = g_face_axes_lut[face_axis][0];
	const unsigned int tangent_y_axis = g_face_axes_lut[face_axis][1];
	Vector3i sample_base = cell_position;
	sample_base[face_axis] += face_sign;

	const auto occupied = [&](const int tangent_x, const int tangent_y) {
		Vector3i sample = sample_base;
		sample[tangent_x_axis] += tangent_x;
		sample[tangent_y_axis] += tangent_y;
		return get_alpha_index(color_func(voxel_buffer[Vector3iUtil::get_zxy_index(sample, block_size)])) != 0;
	};

	const bool side_x_neg = occupied(-1, 0);
	const bool side_x_pos = occupied(1, 0);
	const bool side_y_neg = occupied(0, -1);
	const bool side_y_pos = occupied(0, 1);
	const bool corner_neg_neg = occupied(-1, -1);
	const bool corner_pos_neg = occupied(1, -1);
	const bool corner_neg_pos = occupied(-1, 1);
	const bool corner_pos_pos = occupied(1, 1);
	const auto shade = [](const bool side_x, const bool side_y, const bool corner) {
		const int occlusion = side_x && side_y ? 0 : 3 - int(side_x) - int(side_y) - int(corner);
		return static_cast<uint8_t>(occlusion * 85);
	};
	out_ao[0] = shade(side_x_neg, side_y_neg, corner_neg_neg);
	out_ao[1] = shade(side_x_pos, side_y_neg, corner_pos_neg);
	out_ao[2] = shade(side_x_neg, side_y_pos, corner_neg_pos);
	out_ao[3] = shade(side_x_pos, side_y_pos, corner_pos_pos);
}

inline Color8 apply_vertex_ao(Color8 color, const uint8_t ao, const bool enabled) {
	// COLOR.r is the shader palette slot and COLOR.a is the render-class marker;
	// only the otherwise-unused green component carries AO.
	return enabled ? Color8(color.r, ao, color.b, color.a) : color;
}

template <typename Voxel_T, typename Color_F>
void build_voxel_mesh_as_simple_cubes(
		FixedArray<VoxelMesherCubes::Arrays, VoxelMesherCubes::MATERIAL_COUNT> &out_arrays_per_material,
		const Span<const Voxel_T> voxel_buffer,
		const Vector3i block_size,
		Color_F color_func,
		const bool encode_vertex_ao = false
) {
	//
	ERR_FAIL_COND(
			block_size.x < static_cast<int>(2 * VoxelMesherCubes::PADDING) ||
			block_size.y < static_cast<int>(2 * VoxelMesherCubes::PADDING) ||
			block_size.z < static_cast<int>(2 * VoxelMesherCubes::PADDING)
	);

	const Vector3i min_pos = Vector3iUtil::create(VoxelMesherCubes::PADDING);
	const Vector3i max_pos = block_size - Vector3iUtil::create(VoxelMesherCubes::PADDING);
	const unsigned int row_size = block_size.y;
	const unsigned int deck_size = block_size.x * row_size;

	// Note: voxel buffers are indexed in ZXY order
	FixedArray<uint32_t, Vector3iUtil::AXIS_COUNT> neighbor_offset_d_lut;
	neighbor_offset_d_lut[Vector3i::AXIS_X] = block_size.y;
	neighbor_offset_d_lut[Vector3i::AXIS_Y] = 1;
	neighbor_offset_d_lut[Vector3i::AXIS_Z] = block_size.x * block_size.y;

	FixedArray<uint32_t, VoxelMesherCubes::MATERIAL_COUNT> index_offsets;
	fill(index_offsets, uint32_t(0));

	// For each axis
	for (unsigned int za = 0; za < Vector3iUtil::AXIS_COUNT; ++za) {
		const unsigned int xa = g_face_axes_lut[za][0];
		const unsigned int ya = g_face_axes_lut[za][1];

		// For each deck
		for (unsigned int d = min_pos[za] - VoxelMesherCubes::PADDING; d < (unsigned int)max_pos[za]; ++d) {
			// For each cell of the deck, gather face info
			for (unsigned int fy = min_pos[ya]; fy < (unsigned int)max_pos[ya]; ++fy) {
				for (unsigned int fx = min_pos[xa]; fx < (unsigned int)max_pos[xa]; ++fx) {
					FixedArray<unsigned int, Vector3iUtil::AXIS_COUNT> pos;
					pos[xa] = fx;
					pos[ya] = fy;
					pos[za] = d;

					const unsigned int voxel_index = pos[Vector3i::AXIS_Y] + pos[Vector3i::AXIS_X] * row_size +
							pos[Vector3i::AXIS_Z] * deck_size;

					const Voxel_T raw_color0 = voxel_buffer[voxel_index];
					const Voxel_T raw_color1 = voxel_buffer[voxel_index + neighbor_offset_d_lut[za]];

					const Color8 color0 = color_func(raw_color0);
					const Color8 color1 = color_func(raw_color1);

					// TODO Change this
					const uint8_t ai0 = get_alpha_index(color0);
					const uint8_t ai1 = get_alpha_index(color1);

					Color8 color;
					FaceSide side;
					if (ai0 == ai1) {
						continue;
					} else if (ai0 > ai1) {
						color = color0;
						side = FACE_SIDE_BACK;
					} else {
						color = color1;
						side = FACE_SIDE_FRONT;
					}

					// Commit face to the mesh

					const uint8_t material_index = get_material_index(color);
					VoxelMesherCubes::Arrays &arrays = out_arrays_per_material[material_index];
					const Vector3i cell_position(pos[0], pos[1], pos[2]);
					const int face_sign = side == FACE_SIDE_FRONT ? -1 : 1;
					// The FRONT face belongs to raw_color1, one voxel toward the
					// positive face axis; BACK belongs to raw_color0 at `pos`.
					Vector3i ao_cell_position = cell_position;
					if (side == FACE_SIDE_FRONT) {
						ao_cell_position[za] += 1;
					}

					const int vx0 = fx - VoxelMesherCubes::PADDING;
					const int vy0 = fy - VoxelMesherCubes::PADDING;
					const int vx1 = vx0 + 1;
					const int vy1 = vy0 + 1;

					Vector3f v0;
					v0[xa] = vx0;
					v0[ya] = vy0;
					v0[za] = d;

					Vector3f v1;
					v1[xa] = vx1;
					v1[ya] = vy0;
					v1[za] = d;

					Vector3f v2;
					v2[xa] = vx0;
					v2[ya] = vy1;
					v2[za] = d;

					Vector3f v3;
					v3[xa] = vx1;
					v3[ya] = vy1;
					v3[za] = d;

					Vector3f n;
					n[za] = side == FACE_SIDE_FRONT ? -1 : 1;

					// 2-----3
					// |     |
					// |     |
					// 0-----1

					arrays.positions.push_back(v0);
					arrays.positions.push_back(v1);
					arrays.positions.push_back(v2);
					arrays.positions.push_back(v3);

					// TODO Any way to not need Color anywhere? It's wasteful
					uint8_t ao[4] = { 255, 255, 255, 255 };
					if (encode_vertex_ao) {
						fill_vertex_ao_unchecked(
								voxel_buffer, block_size, ao_cell_position, za, face_sign, color_func, ao);
					}
					const Color colorf0 = apply_vertex_ao(color, ao[0], encode_vertex_ao);
					const Color colorf1 = apply_vertex_ao(color, ao[1], encode_vertex_ao);
					const Color colorf2 = apply_vertex_ao(color, ao[2], encode_vertex_ao);
					const Color colorf3 = apply_vertex_ao(color, ao[3], encode_vertex_ao);
					arrays.colors.push_back(colorf0);
					arrays.colors.push_back(colorf1);
					arrays.colors.push_back(colorf2);
					arrays.colors.push_back(colorf3);

					arrays.normals.push_back(n);
					arrays.normals.push_back(n);
					arrays.normals.push_back(n);
					arrays.normals.push_back(n);

					const unsigned int index_offset = index_offsets[material_index];
					CRASH_COND(za >= 3 || side >= 2);
					const uint8_t *lut = g_indices_lut[za][side];
					for (unsigned int i = 0; i < 6; ++i) {
						arrays.indices.push_back(index_offset + lut[i]);
					}
					index_offsets[material_index] += 4;
				}
			}
		}
	}
}

template <typename Voxel_T, typename Color_F>
void build_voxel_mesh_as_greedy_cubes(
		FixedArray<VoxelMesherCubes::Arrays, VoxelMesherCubes::MATERIAL_COUNT> &out_arrays_per_material,
		const Span<const Voxel_T> voxel_buffer,
		const Vector3i block_size,
		StdVector<uint8_t> &mask_memory_pool,
		Color_F color_func,
		const bool encode_vertex_ao = false
) {
	//
	ERR_FAIL_COND(
			block_size.x < static_cast<int>(2 * VoxelMesherCubes::PADDING) ||
			block_size.y < static_cast<int>(2 * VoxelMesherCubes::PADDING) ||
			block_size.z < static_cast<int>(2 * VoxelMesherCubes::PADDING)
	);

	struct MaskValue {
		Voxel_T color;
		uint8_t side;
		uint8_t ao[4];

		inline bool operator==(const MaskValue &other) const {
			return color == other.color && side == other.side && ao[0] == other.ao[0] && ao[1] == other.ao[1] &&
					ao[2] == other.ao[2] && ao[3] == other.ao[3];
		}

		inline bool operator!=(const MaskValue &other) const {
			return color != other.color || side != other.side || ao[0] != other.ao[0] || ao[1] != other.ao[1] ||
					ao[2] != other.ao[2] || ao[3] != other.ao[3];
		}
	};

	const Vector3i min_pos = Vector3iUtil::create(VoxelMesherCubes::PADDING);
	const Vector3i max_pos = block_size - Vector3iUtil::create(VoxelMesherCubes::PADDING);
	const unsigned int row_size = block_size.y;
	const unsigned int deck_size = block_size.x * row_size;

	// Note: voxel buffers are indexed in ZXY order
	FixedArray<uint32_t, Vector3iUtil::AXIS_COUNT> neighbor_offset_d_lut;
	neighbor_offset_d_lut[Vector3i::AXIS_X] = block_size.y;
	neighbor_offset_d_lut[Vector3i::AXIS_Y] = 1;
	neighbor_offset_d_lut[Vector3i::AXIS_Z] = block_size.x * block_size.y;

	FixedArray<uint32_t, VoxelMesherCubes::MATERIAL_COUNT> index_offsets;
	fill(index_offsets, uint32_t(0));

	// For each axis
	for (unsigned int za = 0; za < Vector3iUtil::AXIS_COUNT; ++za) {
		const unsigned int xa = g_face_axes_lut[za][0];
		const unsigned int ya = g_face_axes_lut[za][1];

		const unsigned int mask_size_x = (max_pos[xa] - min_pos[xa]);
		const unsigned int mask_size_y = (max_pos[ya] - min_pos[ya]);
		const unsigned int mask_area = mask_size_x * mask_size_y;
		// Using the vector as memory pool
		mask_memory_pool.resize(mask_area * sizeof(MaskValue));
		Span<MaskValue> mask(reinterpret_cast<MaskValue *>(mask_memory_pool.data()), 0, mask_area);

		// For each deck
		for (unsigned int d = min_pos[za] - VoxelMesherCubes::PADDING; d < (unsigned int)max_pos[za]; ++d) {
			// For each cell of the deck, gather face info
			for (unsigned int fy = min_pos[ya]; fy < (unsigned int)max_pos[ya]; ++fy) {
				for (unsigned int fx = min_pos[xa]; fx < (unsigned int)max_pos[xa]; ++fx) {
					FixedArray<unsigned int, Vector3iUtil::AXIS_COUNT> pos;
					pos[xa] = fx;
					pos[ya] = fy;
					pos[za] = d;

					const unsigned int voxel_index = pos[Vector3i::AXIS_Y] + pos[Vector3i::AXIS_X] * row_size +
							pos[Vector3i::AXIS_Z] * deck_size;

					const Voxel_T raw_color0 = voxel_buffer[voxel_index];
					const Voxel_T raw_color1 = voxel_buffer[voxel_index + neighbor_offset_d_lut[za]];

					const Color8 color0 = color_func(raw_color0);
					const Color8 color1 = color_func(raw_color1);

					const uint8_t ai0 = get_alpha_index(color0);
					const uint8_t ai1 = get_alpha_index(color1);

					MaskValue mv{};
					if (ai0 == ai1) {
						mv.side = FACE_SIDE_NONE;
					} else if (ai0 > ai1) {
						mv.color = raw_color0;
						mv.side = FACE_SIDE_BACK;
					} else {
						mv.color = raw_color1;
						mv.side = FACE_SIDE_FRONT;
					}

					// Store AO in the mask so greedy merging never crosses a change in
					// baked corner lighting. A disabled path uses a uniform white value,
					// preserving the original greedy merge behavior and vertex colors.
					if (mv.side != FACE_SIDE_NONE) {
						const Vector3i cell_position(pos[0], pos[1], pos[2]);
						const int face_sign = mv.side == FACE_SIDE_FRONT ? -1 : 1;
						Vector3i ao_cell_position = cell_position;
						if (mv.side == FACE_SIDE_FRONT) {
							ao_cell_position[za] += 1;
						}
						if (encode_vertex_ao) {
							fill_vertex_ao_unchecked(
									voxel_buffer, block_size, ao_cell_position, za, face_sign, color_func, mv.ao);
						} else {
							mv.ao[0] = 255;
							mv.ao[1] = 255;
							mv.ao[2] = 255;
							mv.ao[3] = 255;
						}
					}

					mask[(fx - VoxelMesherCubes::PADDING) + (fy - VoxelMesherCubes::PADDING) * mask_size_x] = mv;
				}
			}

			struct L {
				static inline bool is_range_equal(
						const Span<MaskValue> &mask,
						unsigned int xmin,
						unsigned int xmax,
						MaskValue v
				) {
					for (unsigned int x = xmin; x < xmax; ++x) {
						if (mask[x] != v) {
							return false;
						}
					}
					return true;
				}
			};

			// Greedy quads
			for (unsigned int fy = 0; fy < mask_size_y; ++fy) {
				for (unsigned int fx = 0; fx < mask_size_x; ++fx) {
					const unsigned int mask_index = fx + fy * mask_size_x;
					const MaskValue m = mask[mask_index];

					if (m.side == FACE_SIDE_NONE) {
						continue;
					}

					// Check if the next faces are the same along X
					unsigned int rx = fx + 1;
					while (rx < mask_size_x && mask[rx + fy * mask_size_x] == m) {
						++rx;
					}

					// Check if the next rows of faces are the same along Y
					unsigned int ry = fy + 1;
					while (ry < mask_size_y && L::is_range_equal(mask, fx + ry * mask_size_x, rx + ry * mask_size_x, m)
					) {
						++ry;
					}

					// Commit face to the mesh

					const Color8 color8 = color_func(m.color);
					const uint8_t material_index = get_material_index(color8);
					VoxelMesherCubes::Arrays &arrays = out_arrays_per_material[material_index];

					Vector3f v0;
					v0[xa] = fx;
					v0[ya] = fy;
					v0[za] = d;

					Vector3f v1;
					v1[xa] = rx;
					v1[ya] = fy;
					v1[za] = d;

					Vector3f v2;
					v2[xa] = fx;
					v2[ya] = ry;
					v2[za] = d;

					Vector3f v3;
					v3[xa] = rx;
					v3[ya] = ry;
					v3[za] = d;

					Vector3f n;
					n[za] = m.side == FACE_SIDE_FRONT ? -1 : 1;

					// 2-----3
					// |     |
					// |     |
					// 0-----1

					arrays.positions.push_back(v0);
					arrays.positions.push_back(v1);
					arrays.positions.push_back(v2);
					arrays.positions.push_back(v3);

					arrays.colors.push_back(apply_vertex_ao(color8, m.ao[0], encode_vertex_ao));
					arrays.colors.push_back(apply_vertex_ao(color8, m.ao[1], encode_vertex_ao));
					arrays.colors.push_back(apply_vertex_ao(color8, m.ao[2], encode_vertex_ao));
					arrays.colors.push_back(apply_vertex_ao(color8, m.ao[3], encode_vertex_ao));

					arrays.normals.push_back(n);
					arrays.normals.push_back(n);
					arrays.normals.push_back(n);
					arrays.normals.push_back(n);

					const unsigned int index_offset = index_offsets[material_index];
					CRASH_COND(za >= 3 || m.side >= 2);
					const uint8_t *lut = g_indices_lut[za][m.side];
					for (unsigned int i = 0; i < 6; ++i) {
						arrays.indices.push_back(index_offset + lut[i]);
					}
					index_offsets[material_index] += 4;

					for (unsigned int j = fy; j < ry; ++j) {
						for (unsigned int i = fx; i < rx; ++i) {
							mask[i + j * mask_size_x].side = FACE_SIDE_NONE;
						}
					}
				}
			}
		}
	}
}

template <typename Voxel_T, typename Color_F>
void build_voxel_mesh_as_greedy_cubes_atlased(
		FixedArray<VoxelMesherCubes::Arrays, VoxelMesherCubes::MATERIAL_COUNT> &out_arrays_per_material,
		VoxelMesherCubes::GreedyAtlasData &out_greedy_atlas_data,
		const Span<Voxel_T> voxel_buffer,
		const Vector3i block_size,
		StdVector<uint8_t> &mask_memory_pool,
		Color_F color_func
) {
	//
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND(
			block_size.x < static_cast<int>(2 * VoxelMesherCubes::PADDING) ||
			block_size.y < static_cast<int>(2 * VoxelMesherCubes::PADDING) ||
			block_size.z < static_cast<int>(2 * VoxelMesherCubes::PADDING)
	);

	struct MaskValue {
		uint8_t side;
		uint8_t material_index;

		inline bool operator==(const MaskValue &other) const {
			return side == other.side && material_index == other.material_index;
		}

		inline bool operator!=(const MaskValue &other) const {
			return side != other.side || material_index != other.material_index;
		}
	};

	out_greedy_atlas_data.clear();

	const Vector3i min_pos = Vector3iUtil::create(VoxelMesherCubes::PADDING);
	const Vector3i max_pos = block_size - Vector3iUtil::create(VoxelMesherCubes::PADDING);
	const unsigned int row_size = block_size.y;
	const unsigned int deck_size = block_size.x * row_size;

	// Note: voxel buffers are indexed in ZXY order
	FixedArray<uint32_t, Vector3iUtil::AXIS_COUNT> neighbor_offset_d_lut;
	neighbor_offset_d_lut[Vector3i::AXIS_X] = block_size.y;
	neighbor_offset_d_lut[Vector3i::AXIS_Y] = 1;
	neighbor_offset_d_lut[Vector3i::AXIS_Z] = block_size.x * block_size.y;

	FixedArray<uint32_t, VoxelMesherCubes::MATERIAL_COUNT> index_offsets;
	fill(index_offsets, uint32_t(0));

	// For each axis
	for (unsigned int za = 0; za < Vector3iUtil::AXIS_COUNT; ++za) {
		const unsigned int xa = g_face_axes_lut[za][0];
		const unsigned int ya = g_face_axes_lut[za][1];

		const unsigned int mask_size_x = (max_pos[xa] - min_pos[xa]);
		const unsigned int mask_size_y = (max_pos[ya] - min_pos[ya]);
		const unsigned int mask_area = mask_size_x * mask_size_y;
		// Using the vector as memory pool
		const unsigned int mask_memory_size = mask_area * sizeof(MaskValue);
		mask_memory_pool.resize(mask_memory_size + mask_area * sizeof(Color8));
		// `mask` and `colors` are grids covering one deck
		Span<MaskValue> mask(reinterpret_cast<MaskValue *>(mask_memory_pool.data()), 0, mask_area);
		Span<Color8> colors(reinterpret_cast<Color8 *>(mask_memory_pool.data() + mask_memory_size), 0, mask_area);

		// For each deck
		for (unsigned int d = min_pos[za] - VoxelMesherCubes::PADDING; d < (unsigned int)max_pos[za]; ++d) {
			// For each cell of the deck, gather face info
			for (unsigned int fy = min_pos[ya]; fy < (unsigned int)max_pos[ya]; ++fy) {
				for (unsigned int fx = min_pos[xa]; fx < (unsigned int)max_pos[xa]; ++fx) {
					FixedArray<unsigned int, Vector3iUtil::AXIS_COUNT> pos;
					pos[xa] = fx;
					pos[ya] = fy;
					pos[za] = d;

					const unsigned int voxel_index = pos[Vector3i::AXIS_Y] + pos[Vector3i::AXIS_X] * row_size +
							pos[Vector3i::AXIS_Z] * deck_size;

					const Voxel_T raw_color0 = voxel_buffer[voxel_index];
					const Voxel_T raw_color1 = voxel_buffer[voxel_index + neighbor_offset_d_lut[za]];

					const Color8 color0 = color_func(raw_color0);
					const Color8 color1 = color_func(raw_color1);

					const uint8_t ai0 = get_alpha_index(color0);
					const uint8_t ai1 = get_alpha_index(color1);

					MaskValue mv{};
					Color8 color(0, 0, 0, 0);
					if (ai0 == ai1) {
						mv.side = FACE_SIDE_NONE;
					} else if (ai0 > ai1) {
						color = color0;
						mv.side = FACE_SIDE_BACK;
						mv.material_index = get_material_index(color);
					} else {
						color = color1;
						mv.side = FACE_SIDE_FRONT;
						mv.material_index = get_material_index(color);
					}

					const unsigned int mask_index =
							(fx - VoxelMesherCubes::PADDING) + (fy - VoxelMesherCubes::PADDING) * mask_size_x;
					mask[mask_index] = mv;
					colors[mask_index] = color;
				}
			}

			struct L {
				static inline bool is_range_equal(
						const Span<MaskValue> &mask,
						unsigned int xmin,
						unsigned int xmax,
						MaskValue v
				) {
					for (unsigned int x = xmin; x < xmax; ++x) {
						if (mask[x] != v) {
							return false;
						}
					}
					return true;
				}
			};

			// Greedy quads
			for (unsigned int fy = 0; fy < mask_size_y; ++fy) {
				for (unsigned int fx = 0; fx < mask_size_x; ++fx) {
					const unsigned int mask_index = fx + fy * mask_size_x;
					const MaskValue m = mask[mask_index];

					if (m.side == FACE_SIDE_NONE) {
						continue;
					}

					// Check if the next faces are the same along X
					unsigned int rx = fx + 1;
					while (rx < mask_size_x && mask[rx + fy * mask_size_x] == m) {
						++rx;
					}

					// Check if the next rows of faces are the same along Y
					unsigned int ry = fy + 1;
					while (ry < mask_size_y && L::is_range_equal(mask, fx + ry * mask_size_x, rx + ry * mask_size_x, m)
					) {
						++ry;
					}

					// Commit face to the mesh

					const uint8_t material_index = m.material_index;
					VoxelMesherCubes::Arrays &arrays = out_arrays_per_material[material_index];

					Vector3f v0;
					v0[xa] = fx;
					v0[ya] = fy;
					v0[za] = d;

					Vector3f v1;
					v1[xa] = rx;
					v1[ya] = fy;
					v1[za] = d;

					Vector3f v2;
					v2[xa] = fx;
					v2[ya] = ry;
					v2[za] = d;

					Vector3f v3;
					v3[xa] = rx;
					v3[ya] = ry;
					v3[za] = d;

					Vector3f n;
					n[za] = m.side == FACE_SIDE_FRONT ? -1 : 1;

					// 2-----3
					// |     |
					// |     |
					// 0-----1

					arrays.positions.push_back(v0);
					arrays.positions.push_back(v1);
					arrays.positions.push_back(v2);
					arrays.positions.push_back(v3);

					VoxelMesherCubes::GreedyAtlasData::ImageInfo image_info;
					image_info.first_vertex_index = arrays.uvs.size();
					arrays.uvs.resize(arrays.uvs.size() + 4); // Values will be assigned in a second pass

					arrays.normals.push_back(n);
					arrays.normals.push_back(n);
					arrays.normals.push_back(n);
					arrays.normals.push_back(n);

					image_info.size_x = rx - fx;
					image_info.size_y = ry - fy;
					image_info.first_color_index = out_greedy_atlas_data.colors.size();
					out_greedy_atlas_data.colors.resize(
							out_greedy_atlas_data.colors.size() + image_info.size_x * image_info.size_y
					);

					const unsigned int index_offset = index_offsets[material_index];
					CRASH_COND(za >= 3 || m.side >= 2);
					const uint8_t *lut = g_indices_lut[za][m.side];
					for (unsigned int i = 0; i < 6; ++i) {
						arrays.indices.push_back(index_offset + lut[i]);
					}
					index_offsets[material_index] += 4;

					unsigned int im_i = image_info.first_color_index;
					for (unsigned int my = fy; my < ry; ++my) {
						const unsigned int i0 = fx + my * mask_size_x;
						{
							unsigned int i = i0;
							for (unsigned int mx = fx; mx < rx; ++mx) {
								mask[i].side = FACE_SIDE_NONE;
								++i;
							}
						}
						{
							unsigned int i = i0;
							for (unsigned int mx = fx; mx < rx; ++mx) {
								out_greedy_atlas_data.colors[im_i] = colors[i];
								++im_i;
								++i;
							}
						}
						// TODO Actually that code only missed an offset to its destination for each row?
						// Copy colors row by row
						// memcpy(out_greedy_atlas_data.colors.data() + image_info.first_color_index,
						// 		colors.data() + i0,
						// 		image_info.size_x * sizeof(Color8));
					}

					// TODO Optimization: if colors are uniform, we could allocate a shared single pixel instead.
					// This would reduce texture size and packing cost

					image_info.surface_index = material_index;
					out_greedy_atlas_data.images.push_back(image_info);
				}
			}
		}
	}
}

Ref<Image> make_greedy_atlas(
		const VoxelMesherCubes::GreedyAtlasData &atlas_data,
		Span<VoxelMesherCubes::Arrays> surfaces
) {
	//
	ERR_FAIL_COND_V(atlas_data.images.size() == 0, Ref<Image>());
	ZN_PROFILE_SCOPE();

	// Pack rectangles
	StdVector<Vector2i> result_points;
	Vector2i result_size;
	{
		ZN_PROFILE_SCOPE_NAMED("Packing");
		StdVector<Vector2i> sizes;
		sizes.resize(atlas_data.images.size());
		for (unsigned int i = 0; i < atlas_data.images.size(); ++i) {
			const VoxelMesherCubes::GreedyAtlasData::ImageInfo &im = atlas_data.images[i];
			sizes[i] = Vector2i(im.size_x, im.size_y);
		}
		zylann::godot::geometry_2d_make_atlas(to_span(sizes), result_points, result_size);
	}

	// DEBUG
	// Ref<Image> debug_im;
	// debug_im.instance();
	// debug_im->create(result_size.x, result_size.y, false, Image::FORMAT_RGBA8);
	// debug_im->fill(Color(0, 0, 0));
	// for (unsigned int i = 0; i < atlas_data.images.size(); ++i) {
	// 	const Vector2i dst_pos = result_points[i];
	// 	const VoxelMesherCubes::GreedyAtlasData::ImageInfo &im = atlas_data.images[i];
	// 	Ref<Image> tmp;
	// 	tmp.instance();
	// 	tmp->create(im.size_x, im.size_y, false, debug_im->get_format());
	// 	tmp->fill(Color(Math::randf(), Math::randf(), Math::randf()));
	// 	debug_im->blit_rect(tmp, Rect2(0, 0, tmp->get_width(), tmp->get_height()), Vector2f(dst_pos));
	// }
	// debug_im->save_png("debug_atlas_packing.png");

	// Update UVs
	const Vector2f uv_scale(1.f / float(result_size.x), 1.f / float(result_size.y));
	for (unsigned int i = 0; i < atlas_data.images.size(); ++i) {
		const VoxelMesherCubes::GreedyAtlasData::ImageInfo &im = atlas_data.images[i];
		VoxelMesherCubes::Arrays &surface = surfaces[im.surface_index];
		ERR_FAIL_COND_V(im.first_vertex_index + 4 > surface.uvs.size(), Ref<Image>());
		const unsigned int vi = im.first_vertex_index;
		const Vector2f pos(to_vec2f(result_points[i]));
		// 2-----3
		// |     |
		// |     |
		// 0-----1
		surface.uvs[vi] = pos * uv_scale;
		surface.uvs[vi + 1] = (pos + Vector2f(im.size_x, 0)) * uv_scale;
		surface.uvs[vi + 2] = (pos + Vector2f(0, im.size_y)) * uv_scale;
		surface.uvs[vi + 3] = (pos + Vector2f(im.size_x, im.size_y)) * uv_scale;
	}

	// Create image
	PackedByteArray im_data;
	im_data.resize(result_size.x * result_size.y * sizeof(Color8));
	{
		Span<Color8> dst_data = Span<Color8>(reinterpret_cast<Color8 *>(im_data.ptrw()), result_size.x * result_size.y);

		// For all rectangles
		for (unsigned int i = 0; i < atlas_data.images.size(); ++i) {
			const VoxelMesherCubes::GreedyAtlasData::ImageInfo &im = atlas_data.images[i];
			const Vector2i dst_pos = result_points[i];
			Span<const Color8> src_data =
					to_span_from_position_and_size(atlas_data.colors, im.first_color_index, im.size_x * im.size_y);

			// Blit rectangle
			for (unsigned int y = 0; y < im.size_y; ++y) {
				for (unsigned int x = 0; x < im.size_x; ++x) {
					const unsigned int src_i = x + y * im.size_x;
					const unsigned int dst_i = (dst_pos.x + x) + (dst_pos.y + y) * result_size.x;
					dst_data[dst_i] = src_data[src_i];
				}
			}
		}
	}

	Ref<Image> image = Image::create_from_data(result_size.x, result_size.y, false, Image::FORMAT_RGBA8, im_data);
	return image;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VoxelMesherCubes::VoxelMesherCubes() {
	set_padding(PADDING, PADDING);
}

VoxelMesherCubes::~VoxelMesherCubes() {}

VoxelMesherCubes::Cache &VoxelMesherCubes::get_tls_cache() {
	static thread_local Cache cache;
	return cache;
}

void VoxelMesherCubes::build(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	ZN_PROFILE_SCOPE();
	const int channel = VoxelBuffer::CHANNEL_COLOR;
	Cache &cache = get_tls_cache();

	for (unsigned int i = 0; i < cache.arrays_per_material.size(); ++i) {
		Arrays &a = cache.arrays_per_material[i];
		a.clear();
	}

	const VoxelBuffer &voxels = input.voxels;

	// Iterate 3D padded data to extract voxel faces.
	// This is the most intensive job in this class, so all required data should be as fit as possible.

	// The buffer we receive MUST be dense (i.e not compressed, and channels allocated).
	// That means we can use raw pointers to voxel data inside instead of using the higher-level getters,
	// and then save a lot of time.

	if (voxels.get_channel_compression(channel) == VoxelBuffer::COMPRESSION_UNIFORM) {
		// All voxels have the same type.
		// If it's all air, nothing to do. If it's all cubes, nothing to do either.
		return;

	} else if (voxels.get_channel_compression(channel) != VoxelBuffer::COMPRESSION_NONE) {
		// No other form of compression is allowed
		ERR_PRINT("VoxelMesherCubes received unsupported voxel compression");
		return;
	}

	Span<const uint8_t> raw_channel;
	if (!voxels.get_channel_as_bytes_read_only(channel, raw_channel)) {
		// Case supposedly handled before...
		ERR_PRINT("Something wrong happened");
		return;
	}

	const Vector3i block_size = voxels.get_size();
	const VoxelBuffer::Depth channel_depth = voxels.get_channel_depth(channel);

	Parameters params;
	{
		RWLockRead rlock(_parameters_lock);
		params = _parameters;
	}
	// Note, we don't lock the palette because its data has fixed-size

	Ref<Image> atlas_image;

	switch (params.color_mode) {
	case COLOR_RAW:
			switch (channel_depth) {
				case VoxelBuffer::DEPTH_8_BIT:
					if (params.greedy_meshing) {
						build_voxel_mesh_as_greedy_cubes(
								cache.arrays_per_material,
								raw_channel,
								block_size,
								cache.mask_memory_pool,
								Color8::from_u8
						);
					} else {
						build_voxel_mesh_as_simple_cubes(
								cache.arrays_per_material, raw_channel, block_size, Color8::from_u8
						);
					}
					break;

				case VoxelBuffer::DEPTH_16_BIT:
					if (params.greedy_meshing) {
						build_voxel_mesh_as_greedy_cubes(
								cache.arrays_per_material,
								raw_channel.reinterpret_cast_to<const uint16_t>(),
								block_size,
								cache.mask_memory_pool,
								Color8::from_u16
						);
					} else {
						build_voxel_mesh_as_simple_cubes(
								cache.arrays_per_material,
								raw_channel.reinterpret_cast_to<const uint16_t>(),
								block_size,
								Color8::from_u16
						);
					}
					break;

				case VoxelBuffer::DEPTH_32_BIT:
					if (params.greedy_meshing) {
						build_voxel_mesh_as_greedy_cubes(
								cache.arrays_per_material,
								raw_channel.reinterpret_cast_to<const uint32_t>(),
								block_size,
								cache.mask_memory_pool,
								Color8::from_u32
						);
					} else {
						build_voxel_mesh_as_simple_cubes(
								cache.arrays_per_material,
								raw_channel.reinterpret_cast_to<const uint32_t>(),
								block_size,
								Color8::from_u32
						);
					}
					break;

				default:
					ERR_PRINT("Unsupported voxel depth");
					return;
			}
			break;

		case COLOR_MESHER_PALETTE: {
			ERR_FAIL_COND_MSG(params.palette.is_null(), "Palette mode is used but no palette was specified");

			struct GetColorFromPalette {
				VoxelColorPalette &palette;
				Color8 operator()(uint64_t i) const {
					// Note: even though this code may run in a thread, I'm not locking the palette at all because
					// it stores colors in a fixed-size array, and reading the wrong color won't cause any serious
					// problem. It's not supposed to change often in game anyways. If it does, better use shader mode.
					return palette.get_color8(i);
				}
			};
			const GetColorFromPalette get_color_from_palette{ **params.palette };

			switch (channel_depth) {
				case VoxelBuffer::DEPTH_8_BIT:
					if (params.greedy_meshing) {
						if (params.store_colors_in_texture) {
							build_voxel_mesh_as_greedy_cubes_atlased(
									cache.arrays_per_material,
									cache.greedy_atlas_data,
									raw_channel,
									block_size,
									cache.mask_memory_pool,
									get_color_from_palette
						);
							atlas_image =
									make_greedy_atlas(cache.greedy_atlas_data, to_span(cache.arrays_per_material));
						} else {
							build_voxel_mesh_as_greedy_cubes(
									cache.arrays_per_material,
									raw_channel,
									block_size,
									cache.mask_memory_pool,
									get_color_from_palette
							);
						}
					} else {
						build_voxel_mesh_as_simple_cubes(
								cache.arrays_per_material, raw_channel, block_size, get_color_from_palette
						);
					}
					break;

				case VoxelBuffer::DEPTH_16_BIT:
					if (params.greedy_meshing) {
						build_voxel_mesh_as_greedy_cubes(
								cache.arrays_per_material,
								raw_channel.reinterpret_cast_to<const uint16_t>(),
								block_size,
								cache.mask_memory_pool,
								get_color_from_palette
						);
					} else {
						build_voxel_mesh_as_simple_cubes(
								cache.arrays_per_material,
								raw_channel.reinterpret_cast_to<const uint16_t>(),
								block_size,
								get_color_from_palette
						);
					}
					break;

				default:
					ERR_PRINT("Unsupported voxel depth");
					return;
			}
		} break;

		case COLOR_SHADER_PALETTE: {
			ERR_FAIL_COND_MSG(params.palette.is_null(), "Palette mode is used but no palette was specified");
			const uint8_t *render_class_data =
					params.render_class_palette.size() == static_cast<int>(VoxelColorPalette::MAX_COLORS) ?
						params.render_class_palette.ptr() : nullptr;

			struct GetIndexFromPalette {
				VoxelColorPalette &palette;
				uint16_t ignored_color_value;
				const uint8_t *render_class_data;
				Color8 operator()(uint64_t i) const {
					// 阶段 03 fork 定制（palette cube mesher）：
					// cell 编码 0=air、1..256=材质（VoxelBackend）。顶点 R 通道 = 材质槽（cell-1, 0..255），
					// shader 用 R 查 palette lookup texture；air(0) alpha=0 不产生面；
					// 默认实体 alpha=255；CUTOUT/BLENDED 使用不同 marker，分别
					// 进入 native cutout/blended surface。
					uint8_t alpha = 255;
					if (render_class_data != nullptr && i > 0 && i <= VoxelColorPalette::MAX_COLORS) {
						const uint8_t render_class = render_class_data[uint8_t(i - 1)];
						alpha = render_class == 1 ? 254 : render_class == 2 ? 253 : 255;
					}
					return i == 0 || i == ignored_color_value ? Color8(0, 0, 0, 0) :
							Color8(uint8_t(i - 1), 255, 0, alpha);
				}
			};
			const GetIndexFromPalette get_index_from_palette{
					**params.palette, params.ignored_color_value, render_class_data};

			switch (channel_depth) {
				case VoxelBuffer::DEPTH_8_BIT:
					if (params.greedy_meshing) {
						build_voxel_mesh_as_greedy_cubes(
								cache.arrays_per_material,
								raw_channel,
								block_size,
								cache.mask_memory_pool,
								get_index_from_palette,
								params.occlusion_enabled
						);
					} else {
						build_voxel_mesh_as_simple_cubes(
								cache.arrays_per_material,
								raw_channel,
								block_size,
								get_index_from_palette,
								params.occlusion_enabled
						);
					}
					break;

				case VoxelBuffer::DEPTH_16_BIT:
					if (params.greedy_meshing) {
						build_voxel_mesh_as_greedy_cubes(
								cache.arrays_per_material,
								raw_channel.reinterpret_cast_to<const uint16_t>(),
								block_size,
								cache.mask_memory_pool,
								get_index_from_palette,
								params.occlusion_enabled
						);
					} else {
						build_voxel_mesh_as_simple_cubes(
								cache.arrays_per_material,
								raw_channel.reinterpret_cast_to<const uint16_t>(),
								block_size,
								get_index_from_palette,
								params.occlusion_enabled
						);
					}
					break;

				default:
					ERR_PRINT("Unsupported voxel depth");
					return;
			}
		} break;

		default:
			CRASH_NOW();
			break;
	}

	if (input.lod_index > 0) {
		// TODO This is very crude LOD, there will be cracks at the borders.
		// One way would be to not cull faces on chunk borders if any neighbor face is air
		const float lod_scale = 1 << input.lod_index;
		for (unsigned int material_index = 0; material_index < cache.arrays_per_material.size(); ++material_index) {
			Arrays &arrays = cache.arrays_per_material[material_index];
			for (unsigned int i = 0; i < arrays.positions.size(); ++i) {
				arrays.positions[i] *= lod_scale;
			}
		}
	}

	// TODO We could return a single byte array and use Mesh::add_surface down the line?

	for (unsigned int material_index = 0; material_index < MATERIAL_COUNT; ++material_index) {
		const Arrays &arrays = cache.arrays_per_material[material_index];

		if (arrays.positions.size() != 0) {
			Output::Surface surface;
			Array &mesh_arrays = surface.arrays;
			mesh_arrays.resize(Mesh::ARRAY_MAX);

			using namespace zylann::godot;

			{
				PackedVector3Array positions;
				PackedVector3Array normals;
				PackedInt32Array indices;

				copy_to(positions, to_span_const(arrays.positions));
				copy_to(normals, to_span_const(arrays.normals));
				copy_to(indices, to_span_const(arrays.indices));

				mesh_arrays[Mesh::ARRAY_VERTEX] = positions;
				mesh_arrays[Mesh::ARRAY_NORMAL] = normals;
				mesh_arrays[Mesh::ARRAY_INDEX] = indices;

				if (arrays.colors.size() > 0) {
					PackedColorArray colors;
					copy_to(colors, to_span_const(arrays.colors));
					mesh_arrays[Mesh::ARRAY_COLOR] = colors;
				}
				if (arrays.uvs.size() > 0) {
					PackedVector2Array uvs;
					copy_to(uvs, to_span_const(arrays.uvs));
					mesh_arrays[Mesh::ARRAY_TEX_UV] = uvs;
				}
			}

			// surface.collision_enabled = (material_index == MATERIAL_OPAQUE);

			surface.material_index = material_index;
			output.surfaces.push_back(surface);
		}
		//  else {
		// 	// Empty
		// }
	}

	output.primitive_type = Mesh::PRIMITIVE_TRIANGLES;
	output.atlas_image = atlas_image;

	// if (params.store_colors_in_texture) {
	// 	// Don't compress UVs, they need to be precise. Not doing this causes noticeable offsets.
	// 	output.compression_flags = Mesh::ARRAY_COMPRESS_FLAGS_BASE & ~Mesh::ARRAY_FORMAT_TEX_UV;
	// }
	// output.compression_flags = Mesh::ARRAY_COMPRESS_COLOR;
}

void VoxelMesherCubes::set_greedy_meshing_enabled(bool enable) {
	RWLockWrite wlock(_parameters_lock);
	_parameters.greedy_meshing = enable;
}

bool VoxelMesherCubes::is_greedy_meshing_enabled() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.greedy_meshing;
}

void VoxelMesherCubes::set_palette(Ref<VoxelColorPalette> palette) {
	RWLockWrite wlock(_parameters_lock);
	_parameters.palette = palette;
}

Ref<VoxelColorPalette> VoxelMesherCubes::get_palette() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.palette;
}

void VoxelMesherCubes::set_render_class_palette(PackedByteArray palette) {
	ERR_FAIL_COND_MSG(
			palette.size() != 0 && palette.size() != static_cast<int>(VoxelColorPalette::MAX_COLORS),
			"render_class_palette must be empty or contain exactly 256 bytes"
	);
	if (palette.size() == static_cast<int>(VoxelColorPalette::MAX_COLORS)) {
		const uint8_t *data = palette.ptr();
		for (int i = 0; i < palette.size(); ++i) {
			ERR_FAIL_COND_MSG(data[i] > 2, "render_class_palette values must be 0, 1, or 2");
		}
	}
	RWLockWrite wlock(_parameters_lock);
	_parameters.render_class_palette = palette;
}

PackedByteArray VoxelMesherCubes::get_render_class_palette() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.render_class_palette;
}

void VoxelMesherCubes::set_color_mode(ColorMode mode) {
	ERR_FAIL_INDEX(mode, COLOR_MODE_COUNT);
	RWLockWrite wlock(_parameters_lock);
	_parameters.color_mode = mode;
}

VoxelMesherCubes::ColorMode VoxelMesherCubes::get_color_mode() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.color_mode;
}

void VoxelMesherCubes::set_store_colors_in_texture(bool enable) {
	RWLockWrite wlock(_parameters_lock);
	_parameters.store_colors_in_texture = enable;
}

bool VoxelMesherCubes::get_store_colors_in_texture() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.store_colors_in_texture;
}

void VoxelMesherCubes::set_occlusion_enabled(bool enable) {
	RWLockWrite wlock(_parameters_lock);
	_parameters.occlusion_enabled = enable;
}

bool VoxelMesherCubes::get_occlusion_enabled() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.occlusion_enabled;
}

bool VoxelMesherCubes::is_vertex_ao_supported() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.color_mode == COLOR_SHADER_PALETTE && !_parameters.store_colors_in_texture;
}

void VoxelMesherCubes::set_ignored_color_value(int value) {
	ERR_FAIL_COND(value < 0 || value > 65535);
	RWLockWrite wlock(_parameters_lock);
	_parameters.ignored_color_value = static_cast<uint16_t>(value);
}

int VoxelMesherCubes::get_ignored_color_value() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.ignored_color_value;
}

// Ref<Resource> VoxelMesherCubes::duplicate(bool p_subresources) const {
// 	Parameters params;
// 	{
// 		RWLockRead rlock(_parameters_lock);
// 		params = _parameters;
// 	}

// 	if (p_subresources && params.palette.is_valid()) {
// 		params.palette = params.palette->duplicate(true);
// 	}
// 	Ref<VoxelMesherCubes> d;
// 	d.instantiate();
// 	d->_parameters = params;

// 	return d;
// }

int VoxelMesherCubes::get_used_channels_mask() const {
	return (1 << VoxelBuffer::CHANNEL_COLOR);
}

void VoxelMesherCubes::set_material_by_index(Materials id, Ref<Material> material) {
	ERR_FAIL_INDEX(id, int(_materials.size()));
	_materials[id] = material;
}

Ref<Material> VoxelMesherCubes::get_material_by_index(unsigned int i) const {
	ZN_ASSERT_RETURN_V(i < _materials.size(), Ref<Material>());
	return _materials[i];
}

unsigned int VoxelMesherCubes::get_material_index_count() const {
	return _materials.size();
}

void VoxelMesherCubes::_b_set_opaque_material(Ref<Material> material) {
	set_material_by_index(MATERIAL_OPAQUE, material);
}

Ref<Material> VoxelMesherCubes::_b_get_opaque_material() const {
	return get_material_by_index(MATERIAL_OPAQUE);
}

void VoxelMesherCubes::_b_set_transparent_material(Ref<Material> material) {
	// Legacy property alias: transparent used to be the only non-opaque
	// surface, and now maps to the CUTOUT slot.
	set_material_by_index(MATERIAL_CUTOUT, material);
}

Ref<Material> VoxelMesherCubes::_b_get_transparent_material() const {
	return get_material_by_index(MATERIAL_CUTOUT);
}

void VoxelMesherCubes::_b_set_cutout_material(Ref<Material> material) {
	set_material_by_index(MATERIAL_CUTOUT, material);
}

Ref<Material> VoxelMesherCubes::_b_get_cutout_material() const {
	return get_material_by_index(MATERIAL_CUTOUT);
}

void VoxelMesherCubes::_b_set_blended_material(Ref<Material> material) {
	set_material_by_index(MATERIAL_BLENDED, material);
}

Ref<Material> VoxelMesherCubes::_b_get_blended_material() const {
	return get_material_by_index(MATERIAL_BLENDED);
}

Ref<Mesh> VoxelMesherCubes::generate_mesh_from_image(Ref<Image> image, float voxel_size) {
	ZN_PROFILE_SCOPE();
	ZN_ASSERT_RETURN_V(image.is_valid(), Ref<Mesh>());
	ZN_ASSERT_RETURN_V(voxel_size > 0.001f, Ref<Mesh>());
	ZN_ASSERT_RETURN_V_MSG(
			!image->is_compressed(), Ref<Mesh>(), format("Image format not supported: {}", image->get_format())
	);

	// Convert image
	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_COLOR, VoxelBuffer::DEPTH_32_BIT);
	const int im_size_x = image->get_width();
	const int im_size_y = image->get_height();

	// Currently all meshers require pre-padded voxel data...
	voxels.create(
			im_size_x + VoxelMesherCubes::PADDING * 2,
			im_size_y + VoxelMesherCubes::PADDING * 2,
			1 + VoxelMesherCubes::PADDING * 2
	);

	for (int y = 0; y < im_size_y; ++y) {
		for (int x = 0; x < im_size_x; ++x) {
			const Color cf = image->get_pixel(x, y);
			const Color8 c(cf);
			voxels.set_voxel(
					c.to_u32(),
					Vector3i(
							x + VoxelMesherCubes::PADDING,
							// Flip Y axis, since Y goes up in world space, but Y goes down in Image space
							(im_size_y - 1 - y) + VoxelMesherCubes::PADDING,
							VoxelMesherCubes::PADDING
					),
					VoxelBuffer::CHANNEL_COLOR
			);
		}
	}

	// Build mesh

	Ref<VoxelMesherCubes> mesher;
	mesher.instantiate();
	VoxelMesher::Output output;
	VoxelMesher::Input input{ voxels, nullptr, Vector3i(), 0, false };
	mesher->build(output, input);

	if (output.surfaces.size() == 0) {
		return Ref<ArrayMesh>();
	}

	Ref<ArrayMesh> mesh;
	mesh.instantiate();

	const Vector3 centering_offset = -Vector3(im_size_x, im_size_y, 1) / 2.0;

	for (unsigned int i = 0; i < output.surfaces.size(); ++i) {
		using namespace zylann::godot;

		VoxelMesher::Output::Surface &surface = output.surfaces[i];
		Array arrays = surface.arrays;

		if (arrays.is_empty()) {
			continue;
		}

		CRASH_COND(arrays.size() != Mesh::ARRAY_MAX);
		if (!is_surface_triangulated(arrays)) {
			continue;
		}

		offset_surface(arrays, centering_offset);

		if (voxel_size != 1.f) {
			scale_surface(arrays, voxel_size);
		}

		mesh->add_surface_from_arrays(output.primitive_type, arrays, Array(), Dictionary(), output.mesh_flags);
	}

	return mesh;
}

void VoxelMesherCubes::_bind_methods() {
	using Self = VoxelMesherCubes;

	ClassDB::bind_method(D_METHOD("set_greedy_meshing_enabled", "enable"), &Self::set_greedy_meshing_enabled);
	ClassDB::bind_method(D_METHOD("is_greedy_meshing_enabled"), &Self::is_greedy_meshing_enabled);

	ClassDB::bind_method(D_METHOD("set_palette", "palette"), &Self::set_palette);
	ClassDB::bind_method(D_METHOD("get_palette"), &Self::get_palette);
	ClassDB::bind_method(D_METHOD("set_render_class_palette", "palette"), &Self::set_render_class_palette);
	ClassDB::bind_method(D_METHOD("get_render_class_palette"), &Self::get_render_class_palette);

	ClassDB::bind_method(D_METHOD("set_color_mode", "mode"), &Self::set_color_mode);
	ClassDB::bind_method(D_METHOD("get_color_mode"), &Self::get_color_mode);
	ClassDB::bind_method(D_METHOD("set_store_colors_in_texture", "enable"), &Self::set_store_colors_in_texture);
	ClassDB::bind_method(D_METHOD("get_store_colors_in_texture"), &Self::get_store_colors_in_texture);
	ClassDB::bind_method(D_METHOD("set_occlusion_enabled", "enable"), &Self::set_occlusion_enabled);
	ClassDB::bind_method(D_METHOD("get_occlusion_enabled"), &Self::get_occlusion_enabled);
	ClassDB::bind_method(D_METHOD("is_vertex_ao_supported"), &Self::is_vertex_ao_supported);
	ClassDB::bind_method(D_METHOD("set_ignored_color_value", "value"), &Self::set_ignored_color_value);
	ClassDB::bind_method(D_METHOD("get_ignored_color_value"), &Self::get_ignored_color_value);

	ClassDB::bind_method(D_METHOD("set_material_by_index", "id", "material"), &Self::set_material_by_index);
	ClassDB::bind_method(D_METHOD("get_material_index_count"), &Self::get_material_index_count);

	ClassDB::bind_method(D_METHOD("_get_opaque_material"), &Self::_b_get_opaque_material);
	ClassDB::bind_method(D_METHOD("_set_opaque_material", "material"), &Self::_b_set_opaque_material);

	ClassDB::bind_method(D_METHOD("_get_transparent_material"), &Self::_b_get_transparent_material);
	ClassDB::bind_method(D_METHOD("_set_transparent_material", "material"), &Self::_b_set_transparent_material);

	ClassDB::bind_method(D_METHOD("_get_cutout_material"), &Self::_b_get_cutout_material);
	ClassDB::bind_method(D_METHOD("_set_cutout_material", "material"), &Self::_b_set_cutout_material);

	ClassDB::bind_method(D_METHOD("_get_blended_material"), &Self::_b_get_blended_material);
	ClassDB::bind_method(D_METHOD("_set_blended_material", "material"), &Self::_b_set_blended_material);

	ClassDB::bind_static_method(
			Self::get_class_static(),
			D_METHOD("generate_mesh_from_image", "image", "voxel_size"),
			&Self::generate_mesh_from_image
	);

	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "greedy_meshing_enabled"),
			"set_greedy_meshing_enabled",
			"is_greedy_meshing_enabled"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "color_mode", PROPERTY_HINT_ENUM, "Raw,MesherPalette,ShaderPalette"),
			"set_color_mode",
			"get_color_mode"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "ignored_color_value", PROPERTY_HINT_RANGE, "0,65535,1"),
			"set_ignored_color_value",
			"get_ignored_color_value"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "store_colors_in_texture"),
			"set_store_colors_in_texture",
			"get_store_colors_in_texture"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "occlusion_enabled"),
			"set_occlusion_enabled",
			"get_occlusion_enabled"
	);
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT, "palette", PROPERTY_HINT_RESOURCE_TYPE, VoxelColorPalette::get_class_static()
			),
			"set_palette",
			"get_palette"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::PACKED_BYTE_ARRAY, "render_class_palette", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE),
			"set_render_class_palette",
			"get_render_class_palette"
	);

	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT,
					"opaque_material",
					PROPERTY_HINT_RESOURCE_TYPE,
					zylann::godot::MATERIAL_3D_PROPERTY_HINT_STRING
			),
			"_set_opaque_material",
			"_get_opaque_material"
	);
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT,
					"transparent_material",
					PROPERTY_HINT_RESOURCE_TYPE,
					zylann::godot::MATERIAL_3D_PROPERTY_HINT_STRING
			),
			"_set_transparent_material",
			"_get_transparent_material"
	);
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT,
					"cutout_material",
					PROPERTY_HINT_RESOURCE_TYPE,
					zylann::godot::MATERIAL_3D_PROPERTY_HINT_STRING
			),
			"_set_cutout_material",
			"_get_cutout_material"
	);
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT,
					"blended_material",
					PROPERTY_HINT_RESOURCE_TYPE,
					zylann::godot::MATERIAL_3D_PROPERTY_HINT_STRING
			),
			"_set_blended_material",
			"_get_blended_material"
	);

	BIND_ENUM_CONSTANT(MATERIAL_OPAQUE);
	BIND_ENUM_CONSTANT(MATERIAL_CUTOUT);
	BIND_ENUM_CONSTANT(MATERIAL_TRANSPARENT);
	BIND_ENUM_CONSTANT(MATERIAL_BLENDED);
	BIND_ENUM_CONSTANT(MATERIAL_COUNT);

	BIND_ENUM_CONSTANT(COLOR_RAW);
	BIND_ENUM_CONSTANT(COLOR_MESHER_PALETTE);
	BIND_ENUM_CONSTANT(COLOR_SHADER_PALETTE);
}

} // namespace zylann::voxel
