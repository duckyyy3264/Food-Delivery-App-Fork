# Generated by Django 5.1 on 2024-10-10 01:57

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("review", "0011_delivererreviewlike_review_deli_user_id_40f2b9_idx_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="delivererreview",
            name="reply",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="replies",
                to="review.delivererreview",
            ),
        ),
        migrations.AddField(
            model_name="deliveryreview",
            name="reply",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="replies",
                to="review.deliveryreview",
            ),
        ),
        migrations.AddField(
            model_name="dishreview",
            name="reply",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="replies",
                to="review.dishreview",
            ),
        ),
        migrations.AddField(
            model_name="restaurantreview",
            name="reply",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="replies",
                to="review.restaurantreview",
            ),
        ),
    ]