/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#include "scene.h"

void Scene::Add(std::shared_ptr<TriangleModel> model)
{
    this->model_ = TriangleModel::Merge(model_ != nullptr ? model_.get() : nullptr, model.get());
}

void Scene::BuildBvh()
{
    if (model_ == nullptr)
        throw std::invalid_argument("Scene is empty");

    std::vector<std::shared_ptr<Triangle>> triangles;
    for(size_t i = 0; i < model_->Size(); ++i)
    {
        triangles.push_back(std::make_shared<Triangle>(model_->triangles_[i]));
    }

    this->bvh_ = std::unique_ptr<Bvh>(new Bvh(triangles, 0, model_->Size()));
    this->bvh_->volumes_ = triangles;
}