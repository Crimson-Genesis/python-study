# Generated by Django 4.2.6 on 2023-10-19 20:14

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sales", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="sale",
            name="total_price",
            field=models.FloatField(blank=True, null=True),
        ),
    ]
