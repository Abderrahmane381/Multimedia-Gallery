
from flask_restful import Api

from api.resources import (
    ColorFeaturesResource,
    TextureFeaturesResource,
    AllFeaturesResource
)


def init_api(app):
    api = Api(app)

    api.add_resource(
        ColorFeaturesResource,
        "/api/color"
    )

    api.add_resource(
        TextureFeaturesResource,
        "/api/texture"
    )

    api.add_resource(
        AllFeaturesResource,
        "/api/features"
    )

    return api
