# Generated by Django 5.1 on 2024-08-22 12:56

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("order", "0016_alter_restaurantcartdish_quantity"),
    ]

    operations = [
        migrations.AddField(
            model_name="restaurantcartdish",
            name="additional_option",
            field=models.JSONField(blank=True, default=dict, null=True),
        ),
    ]
