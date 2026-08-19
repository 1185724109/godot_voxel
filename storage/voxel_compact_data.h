#ifndef VOXEL_COMPACT_DATA_H
#define VOXEL_COMPACT_DATA_H

#include "../util/containers/std_unordered_map.h"
#include "../util/godot/classes/ref_counted.h"
#include "../util/godot/core/array.h"
#include "../util/godot/core/dictionary.h"
#include "../util/godot/core/packed_arrays.h"
#include "../util/math/vector3i.h"
#include <array>
#include <cstdint>
#include <memory>

namespace zylann::voxel::godot {

// Script-facing authoritative LOD0 storage used by the application.
// Chunks are allocated only after their first non-air cell and released when
// they become empty. Values are logical palette codes: 0=air, 1..256=material.
class VoxelCompactData : public RefCounted {
	GDCLASS(VoxelCompactData, RefCounted)

public:
	static constexpr int CHUNK_SIZE = 16;
	static constexpr int CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
	static constexpr uint16_t MAX_CELL_VALUE = 256;

	bool has_cell(Vector3i pos) const;
	int get_cell(Vector3i pos) const;
	bool set_cell(Vector3i pos, int value);
	bool erase_cell(Vector3i pos);
	void clear();
	int64_t get_cell_count() const;
	int64_t get_chunk_count() const;
	Array get_chunk_ids() const;
	Dictionary get_chunk(Vector3i chunk_pos) const;
	Dictionary get_statistics() const;
	int64_t fill_box(Vector3i min_pos, Vector3i max_pos_exclusive, int value);
	int64_t erase_box(Vector3i min_pos, Vector3i max_pos_exclusive);
	Dictionary export_region_records(Vector3i min_pos, Vector3i max_pos_inclusive) const;
	bool import_records(PackedByteArray records);
	int64_t hash_records(PackedByteArray records) const;

protected:
	static void _bind_methods();

private:
	struct Chunk {
		std::array<uint16_t, CHUNK_VOLUME> cells{};
		std::array<uint64_t, CHUNK_VOLUME / 64> occupancy{};
		uint16_t occupied_count = 0;
	};

	static Vector3i get_chunk_pos(Vector3i pos);
	static Vector3i get_local_pos(Vector3i pos, Vector3i chunk_pos);
	static unsigned int get_index(Vector3i local_pos);
	static bool is_occupied(const Chunk &chunk, unsigned int index);
	static void set_occupied(Chunk &chunk, unsigned int index, bool occupied);
	static uint32_t mix_cell(Vector3i pos, uint16_t value);

	StdUnorderedMap<Vector3i, std::unique_ptr<Chunk>> _chunks;
	int64_t _cell_count = 0;
};

} // namespace zylann::voxel::godot

#endif // VOXEL_COMPACT_DATA_H
