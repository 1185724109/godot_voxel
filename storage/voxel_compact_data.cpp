#include "voxel_compact_data.h"
#include "../util/godot/core/class_db.h"
#include <cstring>

namespace zylann::voxel::godot {

namespace {

inline int floor_div_16(const int value) {
	if (value >= 0) {
		return value / VoxelCompactData::CHUNK_SIZE;
	}
	return -static_cast<int>((-static_cast<int64_t>(value) + VoxelCompactData::CHUNK_SIZE - 1) /
			VoxelCompactData::CHUNK_SIZE);
}

} // namespace

Vector3i VoxelCompactData::get_chunk_pos(const Vector3i pos) {
	return Vector3i(floor_div_16(pos.x), floor_div_16(pos.y), floor_div_16(pos.z));
}

Vector3i VoxelCompactData::get_local_pos(const Vector3i pos, const Vector3i chunk_pos) {
	return pos - chunk_pos * CHUNK_SIZE;
}

unsigned int VoxelCompactData::get_index(const Vector3i local_pos) {
	return static_cast<unsigned int>(local_pos.x + CHUNK_SIZE * (local_pos.y + CHUNK_SIZE * local_pos.z));
}

bool VoxelCompactData::is_occupied(const Chunk &chunk, const unsigned int index) {
	return (chunk.occupancy[index >> 6] & (uint64_t(1) << (index & 63))) != 0;
}

void VoxelCompactData::set_occupied(Chunk &chunk, const unsigned int index, const bool occupied) {
	const uint64_t mask = uint64_t(1) << (index & 63);
	uint64_t &word = chunk.occupancy[index >> 6];
	if (occupied) {
		word |= mask;
	} else {
		word &= ~mask;
	}
}

bool VoxelCompactData::has_cell(const Vector3i pos) const {
	const Vector3i chunk_pos = get_chunk_pos(pos);
	const auto it = _chunks.find(chunk_pos);
	if (it == _chunks.end()) {
		return false;
	}
	return is_occupied(*it->second, get_index(get_local_pos(pos, chunk_pos)));
}

int VoxelCompactData::get_cell(const Vector3i pos) const {
	const Vector3i chunk_pos = get_chunk_pos(pos);
	const auto it = _chunks.find(chunk_pos);
	if (it == _chunks.end()) {
		return 0;
	}
	const unsigned int index = get_index(get_local_pos(pos, chunk_pos));
	return is_occupied(*it->second, index) ? it->second->cells[index] : 0;
}

bool VoxelCompactData::set_cell(const Vector3i pos, const int value) {
	if (value == 0) {
		return erase_cell(pos);
	}
	if (value < 0 || value > MAX_CELL_VALUE) {
		return false;
	}
	const Vector3i chunk_pos = get_chunk_pos(pos);
	auto it = _chunks.find(chunk_pos);
	if (it == _chunks.end()) {
		it = _chunks.emplace(chunk_pos, std::make_unique<Chunk>()).first;
	}
	Chunk &chunk = *it->second;
	const unsigned int index = get_index(get_local_pos(pos, chunk_pos));
	const bool was_occupied = is_occupied(chunk, index);
	if (was_occupied && chunk.cells[index] == value) {
		return false;
	}
	chunk.cells[index] = static_cast<uint16_t>(value);
	if (!was_occupied) {
		set_occupied(chunk, index, true);
		++chunk.occupied_count;
		++_cell_count;
	}
	return true;
}

bool VoxelCompactData::erase_cell(const Vector3i pos) {
	const Vector3i chunk_pos = get_chunk_pos(pos);
	const auto it = _chunks.find(chunk_pos);
	if (it == _chunks.end()) {
		return false;
	}
	Chunk &chunk = *it->second;
	const unsigned int index = get_index(get_local_pos(pos, chunk_pos));
	if (!is_occupied(chunk, index)) {
		return false;
	}
	chunk.cells[index] = 0;
	set_occupied(chunk, index, false);
	--chunk.occupied_count;
	--_cell_count;
	if (chunk.occupied_count == 0) {
		_chunks.erase(it);
	}
	return true;
}

void VoxelCompactData::clear() {
	_chunks.clear();
	_cell_count = 0;
}

int64_t VoxelCompactData::get_cell_count() const {
	return _cell_count;
}

int64_t VoxelCompactData::get_chunk_count() const {
	return _chunks.size();
}

Array VoxelCompactData::get_chunk_ids() const {
	Array ids;
	ids.resize(_chunks.size());
	int i = 0;
	for (const auto &entry : _chunks) {
		ids[i++] = entry.first;
	}
	return ids;
}

Dictionary VoxelCompactData::get_chunk(const Vector3i chunk_pos) const {
	Dictionary result;
	const auto it = _chunks.find(chunk_pos);
	if (it == _chunks.end()) {
		return result;
	}
	const Chunk &chunk = *it->second;
	const Vector3i origin = chunk_pos * CHUNK_SIZE;
	for (unsigned int index = 0; index < CHUNK_VOLUME; ++index) {
		if (!is_occupied(chunk, index)) {
			continue;
		}
		const int x = index % CHUNK_SIZE;
		const int y = (index / CHUNK_SIZE) % CHUNK_SIZE;
		const int z = index / (CHUNK_SIZE * CHUNK_SIZE);
		result[origin + Vector3i(x, y, z)] = chunk.cells[index];
	}
	return result;
}

Dictionary VoxelCompactData::get_statistics() const {
	Dictionary stats;
	stats["cell_count"] = _cell_count;
	stats["chunk_count"] = static_cast<int64_t>(_chunks.size());
	stats["payload_bytes"] = static_cast<int64_t>(_chunks.size() * sizeof(Chunk));
	stats["bytes_per_allocated_chunk"] = static_cast<int64_t>(sizeof(Chunk));
	return stats;
}

int64_t VoxelCompactData::fill_box(const Vector3i min_pos, const Vector3i max_pos_exclusive, const int value) {
	if (value <= 0 || value > MAX_CELL_VALUE || max_pos_exclusive.x <= min_pos.x ||
			max_pos_exclusive.y <= min_pos.y || max_pos_exclusive.z <= min_pos.z) {
		return 0;
	}
	int64_t changed = 0;
	for (int x = min_pos.x; x < max_pos_exclusive.x; ++x) {
		for (int y = min_pos.y; y < max_pos_exclusive.y; ++y) {
			for (int z = min_pos.z; z < max_pos_exclusive.z; ++z) {
				changed += set_cell(Vector3i(x, y, z), value) ? 1 : 0;
			}
		}
	}
	return changed;
}

int64_t VoxelCompactData::erase_box(const Vector3i min_pos, const Vector3i max_pos_exclusive) {
	if (max_pos_exclusive.x <= min_pos.x || max_pos_exclusive.y <= min_pos.y ||
			max_pos_exclusive.z <= min_pos.z) {
		return 0;
	}
	int64_t changed = 0;
	for (int x = min_pos.x; x < max_pos_exclusive.x; ++x) {
		for (int y = min_pos.y; y < max_pos_exclusive.y; ++y) {
			for (int z = min_pos.z; z < max_pos_exclusive.z; ++z) {
				changed += erase_cell(Vector3i(x, y, z)) ? 1 : 0;
			}
		}
	}
	return changed;
}

uint32_t VoxelCompactData::mix_cell(const Vector3i pos, const uint16_t value) {
	uint32_t h = 0x811c9dc5u;
	h = (h ^ (static_cast<uint32_t>(pos.x) + 0x9e3779b9u)) * 0x01000193u;
	h = (h ^ (static_cast<uint32_t>(pos.y) + 0x85ebca77u)) * 0x01000193u;
	h = (h ^ (static_cast<uint32_t>(pos.z) + 0xc2b2ae3du)) * 0x01000193u;
	h = (h ^ value) * 0x01000193u;
	return h;
}

Dictionary VoxelCompactData::export_region_records(
		const Vector3i min_pos, const Vector3i max_pos_inclusive) const {
	PackedByteArray records;
	int64_t count = 0;
	uint64_t hash = 14695981039346656037ull;
	for (int x = min_pos.x; x <= max_pos_inclusive.x; ++x) {
		for (int y = min_pos.y; y <= max_pos_inclusive.y; ++y) {
			for (int z = min_pos.z; z <= max_pos_inclusive.z; ++z) {
				const Vector3i pos(x, y, z);
				const int value = get_cell(pos);
				if (value == 0) {
					continue;
				}
				const int old_size = records.size();
				records.resize(old_size + 14);
				uint8_t *dst = records.ptrw() + old_size;
				const int32_t coords[3] = { pos.x, pos.y, pos.z };
				for (unsigned int axis = 0; axis < 3; ++axis) {
					const uint32_t coordinate = static_cast<uint32_t>(coords[axis]);
					for (unsigned int byte_index = 0; byte_index < 4; ++byte_index) {
						dst[axis * 4 + byte_index] = (coordinate >> (byte_index * 8)) & 0xff;
					}
				}
				dst[12] = value & 0xff;
				dst[13] = (value >> 8) & 0xff;
				hash = (hash ^ mix_cell(pos, static_cast<uint16_t>(value))) * 1099511628211ull;
				++count;
			}
		}
	}
	Dictionary result;
	result["records"] = records;
	result["cell_count"] = count;
	result["hash"] = static_cast<int64_t>(hash);
	return result;
}

bool VoxelCompactData::import_records(const PackedByteArray records) {
	if (records.size() % 14 != 0) {
		return false;
	}
	const uint8_t *src = records.ptr();
	for (int offset = 0; offset < records.size(); offset += 14) {
		int32_t coords[3];
		for (unsigned int axis = 0; axis < 3; ++axis) {
			uint32_t coordinate = 0;
			for (unsigned int byte_index = 0; byte_index < 4; ++byte_index) {
				coordinate |= uint32_t(src[offset + axis * 4 + byte_index]) << (byte_index * 8);
			}
			std::memcpy(&coords[axis], &coordinate, sizeof(coordinate));
		}
		const int value = src[offset + 12] | (int(src[offset + 13]) << 8);
		if (value <= 0 || value > MAX_CELL_VALUE) {
			return false;
		}
	}
	for (int offset = 0; offset < records.size(); offset += 14) {
		int32_t coords[3];
		for (unsigned int axis = 0; axis < 3; ++axis) {
			uint32_t coordinate = 0;
			for (unsigned int byte_index = 0; byte_index < 4; ++byte_index) {
				coordinate |= uint32_t(src[offset + axis * 4 + byte_index]) << (byte_index * 8);
			}
			std::memcpy(&coords[axis], &coordinate, sizeof(coordinate));
		}
		const int value = src[offset + 12] | (int(src[offset + 13]) << 8);
		set_cell(Vector3i(coords[0], coords[1], coords[2]), value);
	}
	return true;
}

int64_t VoxelCompactData::hash_records(const PackedByteArray records) const {
	if (records.size() % 14 != 0) {
		return 0;
	}
	const uint8_t *src = records.ptr();
	uint64_t hash = 14695981039346656037ull;
	for (int offset = 0; offset < records.size(); offset += 14) {
		int32_t coords[3];
		for (unsigned int axis = 0; axis < 3; ++axis) {
			uint32_t coordinate = 0;
			for (unsigned int byte_index = 0; byte_index < 4; ++byte_index) {
				coordinate |= uint32_t(src[offset + axis * 4 + byte_index]) << (byte_index * 8);
			}
			std::memcpy(&coords[axis], &coordinate, sizeof(coordinate));
		}
		const int value = src[offset + 12] | (int(src[offset + 13]) << 8);
		if (value <= 0 || value > MAX_CELL_VALUE) {
			return 0;
		}
		hash = (hash ^ mix_cell(Vector3i(coords[0], coords[1], coords[2]), value)) * 1099511628211ull;
	}
	return static_cast<int64_t>(hash);
}

void VoxelCompactData::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_cell", "pos"), &VoxelCompactData::has_cell);
	ClassDB::bind_method(D_METHOD("get_cell", "pos"), &VoxelCompactData::get_cell);
	ClassDB::bind_method(D_METHOD("set_cell", "pos", "value"), &VoxelCompactData::set_cell);
	ClassDB::bind_method(D_METHOD("erase_cell", "pos"), &VoxelCompactData::erase_cell);
	ClassDB::bind_method(D_METHOD("clear"), &VoxelCompactData::clear);
	ClassDB::bind_method(D_METHOD("get_cell_count"), &VoxelCompactData::get_cell_count);
	ClassDB::bind_method(D_METHOD("get_chunk_count"), &VoxelCompactData::get_chunk_count);
	ClassDB::bind_method(D_METHOD("get_chunk_ids"), &VoxelCompactData::get_chunk_ids);
	ClassDB::bind_method(D_METHOD("get_chunk", "chunk_pos"), &VoxelCompactData::get_chunk);
	ClassDB::bind_method(D_METHOD("get_statistics"), &VoxelCompactData::get_statistics);
	ClassDB::bind_method(D_METHOD("fill_box", "min_pos", "max_pos_exclusive", "value"), &VoxelCompactData::fill_box);
	ClassDB::bind_method(D_METHOD("erase_box", "min_pos", "max_pos_exclusive"), &VoxelCompactData::erase_box);
	ClassDB::bind_method(D_METHOD("export_region_records", "min_pos", "max_pos_inclusive"), &VoxelCompactData::export_region_records);
	ClassDB::bind_method(D_METHOD("import_records", "records"), &VoxelCompactData::import_records);
	ClassDB::bind_method(D_METHOD("hash_records", "records"), &VoxelCompactData::hash_records);
}

} // namespace zylann::voxel::godot
