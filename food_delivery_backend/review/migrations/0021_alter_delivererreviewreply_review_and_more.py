# Generated by Django 5.1 on 2024-10-11 09:43

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("review", "0020_alter_delivererreviewreply_review_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="delivererreviewreply",
            name="review",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="deliverer_replies",
                to="review.delivererreview",
            ),
        ),
        migrations.AlterField(
            model_name="deliveryreviewreply",
            name="review",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="delivery_replies",
                to="review.deliveryreview",
            ),
        ),
        migrations.AlterField(
            model_name="dishreviewreply",
            name="review",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="dish_replies",
                to="review.dishreview",
            ),
        ),
        migrations.AlterField(
            model_name="restaurantreviewreply",
            name="review",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="restaurant_replies",
                to="review.restaurantreview",
            ),
        ),
    ]
