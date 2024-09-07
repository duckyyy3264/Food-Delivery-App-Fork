# Generated by Django 5.1 on 2024-09-01 16:46

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("order", "0033_delivery_restaurant_delivery_user"),
    ]

    operations = [
        migrations.AddField(
            model_name="delivery",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name="delivery",
            name="expired_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="deliveryrequest",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name="deliveryrequest",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, null=True),
        ),
        migrations.AlterField(
            model_name="delivery",
            name="status",
            field=models.CharField(
                choices=[
                    ("FINDING_DRIVER", "Finding Driver"),
                    ("ON_THE_WAY", "On the Way"),
                    ("DELIVERED", "Delivered"),
                    ("CANCELLED", "Cancelled"),
                    ("EXPIRED", "Expired"),
                ],
                default="FINDING_DRIVER",
                max_length=50,
            ),
        ),
        migrations.AlterField(
            model_name="deliveryrequest",
            name="status",
            field=models.CharField(
                choices=[
                    ("PENDING", "Pending"),
                    ("ACCEPTED", "Accepted"),
                    ("DECLINED", "Declined"),
                    ("EXPIRED", "Expired"),
                    ("CANCELLED", "Cancelled"),
                ],
                default="PENDING",
                max_length=20,
            ),
        ),
    ]