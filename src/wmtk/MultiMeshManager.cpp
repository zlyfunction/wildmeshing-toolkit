#include "MultiMeshManager.hpp"
#include <wmtk/simplex/top_dimension_cofaces.hpp>
#include <wmtk/simplex/utils/make_unique.hpp>
#include <wmtk/simplex/utils/tuple_vector_to_homogeneous_simplex_vector.hpp>
#include <wmtk/utils/TupleInspector.hpp>
#include "Mesh.hpp"
#include "SimplicialComplex.hpp"
#include "multimesh/utils/transport_tuple.hpp"
#include "multimesh/utils/tuple_map_attribute_io.hpp"
namespace wmtk {

namespace {} // namespace

Tuple MultiMeshManager::map_tuple_between_meshes(
    const Mesh& source_mesh,
    const Mesh& target_mesh,
    const ConstAccessor<long>& map_accessor,
    const Tuple& source_tuple)
{
    PrimitiveType source_mesh_primitive_type = source_mesh.top_simplex_type();
    PrimitiveType target_mesh_primitive_type = target_mesh.top_simplex_type();
    PrimitiveType min_primitive_type =
        std::min(source_mesh_primitive_type, target_mesh_primitive_type);
    Tuple source_mesh_target_tuple = source_tuple;
    const auto [source_mesh_base_tuple, target_mesh_base_tuple] =
        multimesh::utils::read_tuple_map_attribute(map_accessor, source_tuple);

    if (source_mesh_base_tuple.is_null() || target_mesh_base_tuple.is_null()) {
        return Tuple(); // return null tuple
    }

    if (source_mesh_base_tuple.m_global_cid != source_mesh_target_tuple.m_global_cid) {
        assert(source_mesh_primitive_type > target_mesh_primitive_type);
        const std::vector<Tuple> equivalent_tuples = simplex::top_dimension_cofaces_tuples(
            source_mesh,
            Simplex(target_mesh_primitive_type, source_tuple));
        for (const Tuple& t : equivalent_tuples) {
            if (t.m_global_cid == source_mesh_base_tuple.m_global_cid) {
                source_mesh_target_tuple = t;
                break;
            }
        }
    }

    assert(
        source_mesh_base_tuple.m_global_cid ==
        source_mesh_target_tuple
            .m_global_cid); // make sure that local tuple operations will find a valid sequence

    // we want to repeat switches from source_base_tuple -> source_tuple to
    // target_base _tuple -> return value
    //
    return multimesh::utils::transport_tuple(
        source_mesh_base_tuple,
        source_mesh_target_tuple,
        source_mesh_primitive_type,
        target_mesh_base_tuple,
        target_mesh_primitive_type);
}


MultiMeshManager::MultiMeshManager() = default;

MultiMeshManager::~MultiMeshManager() = default;
MultiMeshManager::MultiMeshManager(const MultiMeshManager& o) = default;
MultiMeshManager::MultiMeshManager(MultiMeshManager&& o) = default;
MultiMeshManager& MultiMeshManager::operator=(const MultiMeshManager& o) = default;
MultiMeshManager& MultiMeshManager::operator=(MultiMeshManager&& o) = default;

bool MultiMeshManager::is_root() const
{
    return m_parent == nullptr;
}

long MultiMeshManager::child_id() const
{
    return m_child_id;
}

std::vector<long> MultiMeshManager::absolute_id() const
{
    if (is_root()) {
        return {};
    } else {
        auto id = m_parent->m_multi_mesh_manager.absolute_id();
        id.emplace_back(m_child_id);
        return id;
    }
}


void MultiMeshManager::register_child_mesh(
    Mesh& my_mesh,
    const std::shared_ptr<Mesh>& child_mesh_ptr,
    const std::vector<std::array<Tuple, 2>>& child_tuple_my_tuple_map)
{
    assert((&my_mesh.m_multi_mesh_manager) == this);
    assert(bool(child_mesh_ptr));

    Mesh& child_mesh = *child_mesh_ptr;

    const PrimitiveType child_primitive_type = child_mesh.top_simplex_type();
    const long new_child_id = long(m_children.size());


    constexpr static long TWO_TUPLE_SIZE = 10;
    constexpr static long DEFAULT_TUPLES_VALUES = -1;
    auto child_to_parent_handle = child_mesh.register_attribute<long>(
        child_to_parent_map_attribute_name(),
        child_primitive_type,
        TWO_TUPLE_SIZE,
        false,
        DEFAULT_TUPLES_VALUES);

    // TODO: make sure that this attribute doesnt already exist
    auto parent_to_child_handle = my_mesh.register_attribute<long>(
        parent_to_child_map_attribute_name(new_child_id),
        child_primitive_type,
        TWO_TUPLE_SIZE,
        false,
        DEFAULT_TUPLES_VALUES);


    auto child_to_parent_accessor = child_mesh.create_accessor(child_to_parent_handle);
    auto parent_to_child_accessor = my_mesh.create_accessor(parent_to_child_handle);


    MultiMeshManager& child_manager = child_mesh.m_multi_mesh_manager;

    // update on child_mesh
    child_manager.map_to_parent_handle = child_to_parent_handle;
    child_manager.m_child_id = new_child_id;
    child_manager.m_parent = &my_mesh;

    // update myself
    m_children.emplace_back(ChildData{child_mesh_ptr, parent_to_child_handle});

    // register maps
    for (const auto& [child_tuple, my_tuple] : child_tuple_my_tuple_map) {
        wmtk::multimesh::utils::symmetric_write_tuple_map_attributes(
            parent_to_child_accessor,
            child_to_parent_accessor,
            my_tuple,
            child_tuple);
    }
}

/*
 * TODO: It is the consumer's responsibility to generate the identity map via a utility function
void MultiMeshManager::register_child_mesh(
    Mesh& my_mesh,
    std::shared_ptr<Mesh> child_mesh,
    const std::vector<long>& child_mesh_simplex_id_map)
{
    PrimitiveType map_type = child_mesh->top_simplex_type();
    std::vector<std::array<Tuple, 2>> child_tuple_my_tuple_map;

    for (long child_cell_id = 0; child_cell_id < long(child_mesh_simplex_id_map.size());
         ++child_cell_id) {
        long parent_cell_id = child_mesh_simplex_id_map[child_cell_id];
        child_tuple_my_tuple_map.push_back(
            {child_mesh->tuple_from_id(map_type, child_cell_id),
             my_mesh.tuple_from_id(map_type, parent_cell_id)});
    }
    register_child_mesh(my_mesh, child_mesh, child_tuple_my_tuple_map);
}
*/

const Mesh& MultiMeshManager::get_root_mesh(const Mesh& my_mesh) const
{
    if (is_root()) {
        return my_mesh;
    } else {
        return m_parent->m_multi_mesh_manager.get_root_mesh(*m_parent);
    }
}
Mesh& MultiMeshManager::get_root_mesh(Mesh& my_mesh)
{
    if (is_root()) {
        return my_mesh;
    } else {
        return m_parent->m_multi_mesh_manager.get_root_mesh(*m_parent);
    }
}
std::vector<std::shared_ptr<Mesh>> MultiMeshManager::get_child_meshes() const
{
    std::vector<std::shared_ptr<Mesh>> ret;
    ret.reserve(m_children.size());
    for (const ChildData& cd : m_children) {
        ret.emplace_back(cd.mesh);
    }
    return ret;
}

std::vector<Simplex>
MultiMeshManager::map(const Mesh& my_mesh, const Mesh& other_mesh, const Simplex& my_simplex) const
{
    const auto ret_tups = map_tuples(my_mesh, other_mesh, my_simplex);
    return simplex::utils::tuple_vector_to_homogeneous_simplex_vector(
        ret_tups,
        my_simplex.primitive_type());
}
std::vector<Tuple> MultiMeshManager::map_tuples(
    const Mesh& my_mesh,
    const Mesh& other_mesh,
    const Simplex& my_simplex) const
{
    const PrimitiveType pt = my_simplex.primitive_type();
    assert((&my_mesh.m_multi_mesh_manager) == this);
    std::vector<Tuple> equivalent_tuples =
        simplex::top_dimension_cofaces_tuples(my_mesh, my_simplex);
    // MultiMeshMapVisitor visitor(my_mesh, other_mesh);
    // const auto my_id = absolute_id(); someday could be used to map down
    const auto other_id = other_mesh.absolute_multi_mesh_id();

    // get a root tuple by converting the tuple up parent meshes until root is found
    Tuple cur_tuple = my_simplex.tuple();
    const Mesh* cur_mesh = &my_mesh;
    while (cur_mesh !=
           nullptr) { // cur_mesh == nullptr if we just walked past the root node so we stop
        cur_tuple = cur_mesh->m_multi_mesh_manager.map_tuple_to_parent_tuple(*cur_mesh, cur_tuple);
        cur_mesh = cur_mesh->m_multi_mesh_manager.m_parent;
    }

    // bieng lazy about how i set cur_mesh to nullptr above - could simplify the loop to optimize
    cur_mesh = &get_root_mesh(other_mesh);


    // note that (cur_mesh, tuples) always match (i.e tuples are tuples from cur_mesh)
    std::vector<Tuple> tuples;
    tuples.emplace_back(cur_tuple);

    for (auto it = other_id.rbegin(); it != other_id.rend(); ++it) {
        // get the select ID from the child map
        long child_index = *it;
        const ChildData& cd = cur_mesh->m_multi_mesh_manager.m_children.at(child_index);

        // for every tuple we have try to collect all versions
        std::vector<Tuple> new_tuples;
        for (const Tuple& t : tuples) {
            // get new tuples for every version that exists
            std::vector<Tuple> n =
                cur_mesh->m_multi_mesh_manager.map_to_child_tuples(*cur_mesh, cd, Simplex(pt, t));
            // append to the current set of new tuples
            new_tuples.insert(new_tuples.end(), n.begin(), n.end());
        }
        // update the (mesh,tuples) pair
        tuples = std::move(new_tuples);
        cur_mesh = cd.mesh.get();

        // the front id of the current mesh should be the child index from this iteration
        assert(cur_mesh->m_multi_mesh_manager.m_child_id == child_index);
    }

    // visitor.map(equivalent_tuples, my_simplex.primitive_type());

    return tuples;
}

Simplex MultiMeshManager::map_to_root(const Mesh& my_mesh, const Simplex& my_simplex) const
{
    return Simplex(my_simplex.primitive_type(), map_to_root_tuple(my_mesh, my_simplex));
}

Tuple MultiMeshManager::map_to_root_tuple(const Mesh& my_mesh, const Simplex& my_simplex) const
{
    return map_tuple_to_root_tuple(my_mesh, my_simplex.tuple());
}
Tuple MultiMeshManager::map_tuple_to_root_tuple(const Mesh& my_mesh, const Tuple& my_tuple) const
{
    if (is_root()) {
        return my_tuple;
    } else {
        return map_tuple_to_root_tuple(*m_parent, map_tuple_to_parent_tuple(my_mesh, my_tuple));
    }
}


Simplex MultiMeshManager::map_to_parent(const Mesh& my_mesh, const Simplex& my_simplex) const
{
    return Simplex(
        my_simplex.primitive_type(),
        map_tuple_to_parent_tuple(my_mesh, my_simplex.tuple()));
}
Tuple MultiMeshManager::map_to_parent_tuple(const Mesh& my_mesh, const Simplex& my_simplex) const
{
    return map_tuple_to_parent_tuple(my_mesh, my_simplex.tuple());
}

Tuple MultiMeshManager::map_tuple_to_parent_tuple(const Mesh& my_mesh, const Tuple& my_tuple) const
{
    assert((&my_mesh.m_multi_mesh_manager) == this);
    assert(!is_root());

    const Mesh& parent_mesh = *m_parent;

    const auto& map_handle = map_to_parent_handle;
    // assert(!map_handle.is_null());

    auto map_accessor = my_mesh.create_accessor(map_handle);
    return map_tuple_between_meshes(my_mesh, parent_mesh, map_accessor, my_tuple);
}
std::vector<Tuple> MultiMeshManager::map_to_child_tuples(
    const Mesh& my_mesh,
    const ChildData& child_data,
    const Simplex& my_simplex) const
{
    assert((&my_mesh.m_multi_mesh_manager) == this);

    const Mesh& child_mesh = *child_data.mesh;
    if (child_mesh.top_simplex_type() < my_simplex.primitive_type()) {
        return {};
    }
    const auto map_handle = child_data.map_handle;
    // we will overwrite these tuples inline with the mapped ones while running down the map
    // functionalities
    std::vector<Tuple> tuples = simplex::top_dimension_cofaces_tuples(my_mesh, my_simplex);

    auto map_accessor = my_mesh.create_accessor(map_handle);
    for (Tuple& tuple : tuples) {
        tuple = map_tuple_between_meshes(my_mesh, child_mesh, map_accessor, tuple);
    }
    tuples.erase(
        std::remove_if(
            tuples.begin(),
            tuples.end(),
            [](const Tuple& t) -> bool { return t.is_null(); }),
        tuples.end());
    tuples =
        wmtk::simplex::utils::make_unique_tuples(child_mesh, tuples, my_simplex.primitive_type());

    return tuples;
}

std::vector<Tuple> MultiMeshManager::map_to_child_tuples(
    const Mesh& my_mesh,
    const Mesh& child_mesh,
    const Simplex& my_simplex) const
{
    return map_to_child_tuples(my_mesh, child_mesh.m_multi_mesh_manager.child_id(), my_simplex);
}

std::vector<Tuple> MultiMeshManager::map_to_child_tuples(
    const Mesh& my_mesh,
    long child_id,
    const Simplex& my_simplex) const
{
    // this is just to do a little redirection for simpplifying map_to_child (and potentially for a
    // visitor pattern)
    return map_to_child_tuples(my_mesh, m_children.at(child_id), my_simplex);
}

std::vector<Simplex> MultiMeshManager::map_to_child(
    const Mesh& my_mesh,
    const Mesh& child_mesh,
    const Simplex& my_simplex) const
{
    auto tuples = map_to_child_tuples(my_mesh, child_mesh, my_simplex);
    return simplex::utils::tuple_vector_to_homogeneous_simplex_vector(
        tuples,
        my_simplex.primitive_type());
}


std::vector<std::array<Tuple, 2>> MultiMeshManager::same_simplex_dimension_surjection(
    const Mesh& parent,
    const Mesh& child,
    const std::vector<long>& parent_simplices)
{
    PrimitiveType primitive_type = parent.top_simplex_type();
    assert(primitive_type == child.top_simplex_type());

    long size = child.capacity(primitive_type);
    assert(size == long(parent_simplices.size()));
    std::vector<std::array<Tuple, 2>> ret;
    ret.reserve(size);

    auto parent_flag_accessor = parent.get_const_flag_accessor(primitive_type);
    auto child_flag_accessor = child.get_const_flag_accessor(primitive_type);

    for (long index = 0; index < size; ++index) {
        const Tuple ct = child.tuple_from_id(primitive_type, index);
        const Tuple pt = parent.tuple_from_id(primitive_type, parent_simplices.at(index));
        if ((parent_flag_accessor.const_scalar_attribute(pt) & 1) == 0) {
            continue;
        }
        if ((child_flag_accessor.const_scalar_attribute(ct) & 1) == 0) {
            continue;
        }

        ret.emplace_back(std::array<Tuple, 2>{{ct, pt}});
    }
    return ret;
}

std::string MultiMeshManager::parent_to_child_map_attribute_name(long index)
{
    return fmt::format("map_to_child_{}", index);
}
std::array<attribute::MutableAccessor<long>, 2> MultiMeshManager::get_map_accessors(
    Mesh& my_mesh,
    ChildData& c)
{
    Mesh& child_mesh = *c.mesh;
    const auto& child_to_parent_handle = child_mesh.m_multi_mesh_manager.map_to_parent_handle;
    const auto& parent_to_child_handle = c.map_handle;


    return std::array<attribute::MutableAccessor<long>, 2>{
        {my_mesh.create_accessor(parent_to_child_handle),
         child_mesh.create_accessor(child_to_parent_handle)}};
}
std::array<attribute::ConstAccessor<long>, 2> MultiMeshManager::get_map_const_accessors(
    const Mesh& my_mesh,
    const ChildData& c) const
{
    const Mesh& child_mesh = *c.mesh;
    const auto& child_to_parent_handle = child_mesh.m_multi_mesh_manager.map_to_parent_handle;
    const auto& parent_to_child_handle = c.map_handle;


    return std::array<attribute::ConstAccessor<long>, 2>{
        {my_mesh.create_const_accessor(parent_to_child_handle),
         child_mesh.create_const_accessor(child_to_parent_handle)}};
}
std::string MultiMeshManager::child_to_parent_map_attribute_name()
{
    return "map_to_parent";
}

void MultiMeshManager::update_map_tuple_hashes(
    Mesh& my_mesh,
    PrimitiveType primitive_type,
    const std::vector<std::tuple<long, std::vector<Tuple>>>& simplices_to_update,
    const std::vector<std::tuple<long, std::array<long, 2>>>& split_cell_maps)
{
    // spdlog::info(
    //     "Update map on [{}] for {} (have {})",
    //     fmt::join(my_mesh.absolute_multi_mesh_id(), ","),
    //     primitive_type_name(primitive_type),
    //     simplices_to_update.size());
    //  for (const auto& [gid, tups] : simplices_to_update) {
    //      spdlog::info(
    //          "[{}] Trying to update {}",
    //          fmt::join(my_mesh.absolute_multi_mesh_id(), ","),
    //          gid);
    //  }
    //   parent cells might have been destroyed
    //

    const PrimitiveType parent_primitive_type = my_mesh.top_simplex_type();

    auto parent_hash_accessor = my_mesh.get_const_cell_hash_accessor();
    auto parent_flag_accessor = my_mesh.get_const_flag_accessor(primitive_type);
    // auto& update_tuple = [&](const auto& flag_accessor, Tuple& t) -> bool {
    //     if(acc.index_access().
    // };


    // go over every child mesh and try to update their hashes
    for (auto& child_data : children()) {
        auto& child_mesh = *child_data.mesh;
        // ignore ones whos map are the wrong dimension
        if (child_mesh.top_simplex_type() != primitive_type) {
            continue;
        }
        // spdlog::info(
        //     "[{}->{}] Doing a child mesh",
        //     fmt::join(my_mesh.absolute_multi_mesh_id(), ","),
        //     fmt::join(child_mesh.absolute_multi_mesh_id(), ","));
        //  get accessors to the maps
        auto maps = get_map_accessors(my_mesh, child_data);
        auto& [parent_to_child_accessor, child_to_parent_accessor] = maps;

        auto child_flag_accessor = child_mesh.get_const_flag_accessor(primitive_type);
        auto child_hash_accessor = child_mesh.get_const_cell_hash_accessor();


        // for (const auto& t : my_mesh.get_all(primitive_type)) {
        //     spdlog::warn("{}", my_mesh.id(t, primitive_type));
        // }
        for (const auto& [original_parent_gid, equivalent_parent_tuples] : simplices_to_update) {
            const char parent_flag = Mesh::get_index_access(parent_flag_accessor)
                                         .const_scalar_attribute(original_parent_gid);
            bool exists = 1 == (parent_flag & 1);
            if (!exists) {
                continue;
            }
            // spdlog::info(
            //     "[{}->{}] Trying to update {}",
            //     fmt::join(my_mesh.absolute_multi_mesh_id(), ","),
            //     fmt::join(child_mesh.absolute_multi_mesh_id(), ","),
            //     original_parent_gid);
            //  read off the original map's data
            auto parent_to_child_data = Mesh::get_index_access(parent_to_child_accessor)
                                            .const_vector_attribute(original_parent_gid);

            // read off the data in the Tuple format
            Tuple parent_tuple =
                wmtk::multimesh::utils::vector5_to_tuple(parent_to_child_data.head<5>());
            Tuple child_tuple =
                wmtk::multimesh::utils::vector5_to_tuple(parent_to_child_data.tail<5>());


            // If the parent tuple is invalid then there was no map so we can try the next cell
            if (parent_tuple.is_null()) {
                continue;
            }

            // navigate parent_original_sharer -> parent_tuple
            // then take parent_new_sharer -> parent_tuple
            parent_tuple = my_mesh.resurrect_tuple(parent_tuple, parent_hash_accessor);
            child_tuple = child_mesh.resurrect_tuple(child_tuple, child_hash_accessor);

            // for(const auto& [old_cid, new_cids]: split_cell_maps) {
            //     if(old_cid == original_parent_gid) {

            //    }
            //}

            // Find a valid representation of this simplex representation of the original tupl
            Tuple old_tuple;
            std::optional<Tuple> old_tuple_opt = find_tuple_from_gid(
                my_mesh,
                primitive_type,
                equivalent_parent_tuples,
                original_parent_gid);
            assert(old_tuple_opt.has_value());
            Simplex old_simplex(primitive_type, old_tuple_opt.value());

            std::optional<Tuple> new_parent_shared_opt = find_valid_tuple(
                my_mesh,
                old_simplex,
                original_parent_gid,
                equivalent_parent_tuples,
                split_cell_maps);


            assert(new_parent_shared_opt.has_value());

            Tuple new_parent_tuple_shared = new_parent_shared_opt.value();
            // spdlog::info(
            //     "{} => {} ==> {}",
            //     wmtk::utils::TupleInspector::as_string(old_simplex.tuple()),
            //     wmtk::utils::TupleInspector::as_string(parent_tuple),
            //     wmtk::utils::TupleInspector::as_string(child_tuple));

            parent_tuple = wmtk::multimesh::utils::transport_tuple(
                old_simplex.tuple(),
                parent_tuple,
                primitive_type,
                new_parent_tuple_shared,
                primitive_type);
            parent_tuple = my_mesh.resurrect_tuple(parent_tuple, parent_hash_accessor);
            assert(my_mesh.is_valid_slow(parent_tuple));
            assert(child_mesh.is_valid_slow(child_tuple));


            wmtk::multimesh::utils::symmetric_write_tuple_map_attributes(
                parent_to_child_accessor,
                child_to_parent_accessor,
                parent_tuple,
                child_tuple);
        }
    }
}
std::optional<Tuple> MultiMeshManager::find_valid_tuple(
    Mesh& my_mesh,
    const Simplex& old_simplex,
    const long old_gid,
    const std::vector<Tuple>& equivalent_parent_tuples,
    const std::vector<std::tuple<long, std::array<long, 2>>>& split_cell_maps) const
{
    // if old gid was one of the originals then do tuple
    // otherwise just find some random tuple that still exists

    std::optional<Tuple> split_attempt = find_valid_tuple_from_split(
        my_mesh,
        old_simplex,
        old_gid,
        equivalent_parent_tuples,
        split_cell_maps);
    if (!split_attempt.has_value()) {
        split_attempt = find_valid_tuple_from_alternatives(
            my_mesh,
            old_simplex.primitive_type(),
            equivalent_parent_tuples);
    }

    return split_attempt;
}


std::optional<Tuple> MultiMeshManager::find_valid_tuple_from_alternatives(
    Mesh& my_mesh,
    PrimitiveType primitive_type,
    const std::vector<Tuple>& tuple_alternatives) const
{
    auto parent_flag_accessor = my_mesh.get_const_flag_accessor(primitive_type);
    // find a new sharer by finding a tuple that exists
    auto it = std::find_if(
        tuple_alternatives.begin(),
        tuple_alternatives.end(),
        [&](const Tuple& t) -> bool {
            return 1 == (Mesh::get_index_access(parent_flag_accessor)
                             .scalar_attribute(wmtk::utils::TupleInspector::global_cid(t)) &
                         1);
        });
    if (it != tuple_alternatives.end()) {
        return *it;
    } else {
        return std::optional<Tuple>{};
    }
}

std::optional<Tuple> MultiMeshManager::find_valid_tuple_from_split(
    Mesh& my_mesh,
    const Simplex& old_simplex,
    const long old_simplex_gid,
    const std::vector<Tuple>& tuple_alternatives,
    const std::vector<std::tuple<long, std::array<long, 2>>>& split_cell_maps) const
{
    const Tuple& old_tuple = old_simplex.tuple();
    const PrimitiveType primitive_type = old_simplex.primitive_type();

    for (const auto& [old_cid, new_cids] : split_cell_maps) {
        if (old_cid != old_simplex_gid) {
            continue;
        }

        auto old_tuple_opt =
            find_tuple_from_gid(my_mesh, primitive_type, tuple_alternatives, old_cid);

        assert(old_tuple_opt.has_value());

        const Tuple& old_cid_tuple = old_tuple_opt.value();
        for (const long new_cid : new_cids) {
            // try seeing if we get the right gid by shoving in the new face id
            Tuple tuple(
                wmtk::utils::TupleInspector::local_vid(old_cid_tuple),
                wmtk::utils::TupleInspector::local_eid(old_cid_tuple),
                wmtk::utils::TupleInspector::local_fid(old_cid_tuple),
                new_cid,
                my_mesh.get_cell_hash_slow(new_cid));


            if (my_mesh.is_valid_slow(tuple) &&
                old_simplex_gid == my_mesh.id(tuple, primitive_type)) {
                return tuple;
            }
        }
    }
    return std::optional<Tuple>{};
}

std::optional<Tuple> MultiMeshManager::find_tuple_from_gid(
    const Mesh& my_mesh,
    PrimitiveType primitive_type,
    const std::vector<Tuple>& tuples,
    long gid)
{
    // spdlog::info("Finding gid {}", gid);
    // for(const auto& t: tuples) {
    //     spdlog::info("Found {}", my_mesh.id(t, primitive_type));

    //}

    auto it = std::find_if(tuples.begin(), tuples.end(), [&](const Tuple& t) -> bool {
        return gid == my_mesh.id(t, primitive_type);
    });
    if (it == tuples.end()) {
        // spdlog::info("failed to find tuple");
        return std::optional<Tuple>{};
    } else {
        // spdlog::info("got tuple");
        return *it;
    }
}
long MultiMeshManager::child_global_cid(
    const attribute::ConstAccessor<long>& parent_to_child,
    long parent_gid)
{
    // look at src/wmtk/multimesh/utils/tuple_map_attribute_io.cpp to see what index global_cid gets mapped to)
    // 5 is the size of a tuple is 5 longs, global_cid currently gets written to position 3
    return Mesh::get_index_access(parent_to_child).vector_attribute(parent_gid)(5 + 3);
}
long MultiMeshManager::parent_global_cid(
    const attribute::ConstAccessor<long>& child_to_parent,
    long child_gid)
{
    // look at src/wmtk/multimesh/utils/tuple_map_attribute_io.cpp to see what index global_cid gets mapped to)
    // 5 is the size of a tuple is 5 longs, global_cid currently gets written to position 3
    return Mesh::get_index_access(child_to_parent).vector_attribute(child_gid)(5 + 3);
}
} // namespace wmtk