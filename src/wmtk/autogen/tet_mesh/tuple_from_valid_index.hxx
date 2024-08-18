#pragma once
// DO NOT MODIFY, autogenerated from the /scripts directory
#include "autogenerated_tables.hpp"
#include "tuple_from_valid_index.hpp"

namespace wmtk::autogen::tet_mesh {
inline Tuple tuple_from_valid_index(int64_t global_cid, int8_t valid_tuple_index)
{
    const auto& [lvid, leid, lfid] = auto_valid_tuples[valid_tuple_index];

    return Tuple(lvid, leid, lfid, global_cid);
}

} // namespace wmtk::autogen::tet_mesh
