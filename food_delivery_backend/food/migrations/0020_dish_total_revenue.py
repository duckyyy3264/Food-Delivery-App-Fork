# Generated by Django 5.1 on 2024-10-04 07:05

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("food", "0019_dish_food_dish_categor_17c558_idx"),
    ]

    operations = [
        migrations.AddField(
            model_name="dish",
            name="total_revenue",
            field=models.DecimalField(
                blank=True, decimal_places=2, default=0, max_digits=12, null=True
            ),
        ),
    ]
